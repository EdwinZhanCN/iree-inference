"""Replace quantized ops of an imported esp-dl-quantized model with dispatches
to esp-dl kernels.

Runs on IREE's input-phase IR (iree-compile --compile-to=input). An imported
QDQ ONNX model is there float linalg ops between
iree_linalg_ext.dequantize_affine and quantize_affine. For each quantize_affine
that ends a quantized convolution or matrix product, this script reads the
int8 input, the int8 weights, int32 bias and scales, checks them against
esp-dl's scheme, and replaces the result with a call to a function that
transposes to channels-last and dispatches the import espdl_conv2d_s8.

Recognized, ending in quantize_affine (optionally after a ReLU):
  conv_2d_nchw_fchw, depthwise_conv_2d_nchw_chw, conv_1d_ncw_fcw
      input: dequantize_affine(x), optionally through tensor.pad with 0
      weights: dequantize_affine(w) (depthwise: then collapse_shape)
      init: broadcast(dequantize_affine(bias)) or fill 0
  batch_matmul with batch 1 and constant weights
      input: dequantize_affine(x), optionally transposed [0, 2, 1]
      weights: broadcast of dequantize_affine(w) over the batch
      init: fill 0
Anything else, including scales that are not powers of two, is left to IREE.

The import is resolved at run time: host_plugin.c on the host, the esp-dl
adapter on ESP32-P4. Its contract is espdl_ref.h.

usage: espdl_rewrite.py <in.mlir> <out.mlir>
"""

import sys

import numpy as np
from iree.compiler import ir

# Field order of espdl_ref.h's ESPDL_CONV_* parameters.
PARAMS = ["H", "W", "C", "O", "KH", "KW", "SH", "SW", "PT", "PL", "PB", "PR",
          "DH", "DW", "G", "OH", "OW", "RELU", "X_EXP", "Y_EXP"]


class Skip(Exception):
    """The op does not match; leave it to IREE."""


# ---- IR helpers ---------------------------------------------------------------

def walk(op):
    for region in op.regions:
        for block in region.blocks:
            for o in block.operations:
                yield o.operation
                yield from walk(o.operation)


def producer(value, *names):
    owner = value.owner
    if isinstance(owner, ir.Block):
        raise Skip("block argument")
    op = owner.operation
    if names and op.name not in names:
        raise Skip(f"expected {names}, got {op.name}")
    return op


def attr_str(op, name):
    return str(op.attributes[name]) if name in op.attributes else None


def i64_array(op, name):
    a = op.attributes[name]
    try:
        return [int(v) for v in ir.DenseI64ArrayAttr(a)]
    except (TypeError, ValueError):
        return [int(v) for v in np.array(a).reshape(-1)]


def shape(value):
    return list(ir.RankedTensorType(value.type).shape)


def const(value):
    """The value of an arith.constant as a numpy array (scalars: 0-d)."""
    op = producer(value, "arith.constant")
    a = op.attributes["value"]
    if isinstance(a, ir.FloatAttr):
        return np.array(a.value, dtype=np.float32)
    if isinstance(a, ir.IntegerAttr):
        return np.array(a.value)
    arr = np.array(a)
    full = shape(value)
    return np.broadcast_to(arr, full).copy() if list(arr.shape) != full else arr


def dequant(value, bits=8):
    """(int tensor value, scale array) of a dequantize_affine with no zero point."""
    op = producer(value, "iree_linalg_ext.dequantize_affine")
    lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    if len(op.operands) != 3 or attr_str(op, "quant_min") != f"{lo} : i64" \
            or attr_str(op, "quant_max") != f"{hi} : i64":
        raise Skip("dequantize_affine with zero point or other range")
    return op.operands[0], const(op.operands[1])


def is_relu(op):
    if op.name != "linalg.generic" or len(op.operands) != 2:
        return False
    body = [o.operation for o in op.regions[0].blocks[0].operations]
    if [o.name for o in body] != ["arith.cmpf", "arith.select", "linalg.yield"]:
        return False
    cmp, sel = body[0], body[1]
    try:
        zero = const(cmp.operands[1])
    except Skip:
        return False
    return (attr_str(cmp, "predicate") in ("8 : i64", "2 : i64")  # ugt, ogt
            and float(zero) == 0.0 and sel.operands[2] == cmp.operands[1])


def exponent(scale):
    """Integer exponents of power-of-two scales, or Skip."""
    s = np.asarray(scale, dtype=np.float64)
    m, e = np.frexp(s)
    if not np.all(m == 0.5):
        raise Skip("scale not a power of two")
    return (e - 1).astype(np.int64)


# ---- Matching -------------------------------------------------------------------

def match_input(value):
    """x int8 value and its exponent, and pads (top, left, bottom, right)."""
    pads = [0, 0, 0, 0]
    op = producer(value)
    if op.name == "tensor.pad":
        low, high = i64_array(op, "static_low"), i64_array(op, "static_high")
        if any(low[:2]) or any(high[:2]) or op.operands[1:]:
            raise Skip("pad on batch/channel or dynamic")
        y = op.regions[0].blocks[0].operations[0].operation
        if float(const(y.operands[0])) != 0.0:
            raise Skip("pad value not 0")
        sp = len(low) - 2
        pads = [low[2], low[3], high[2], high[3]] if sp == 2 else [0, low[2], 0, high[2]]
        value = op.operands[0]
    xq, sx = dequant(value)
    if sx.ndim != 0:
        raise Skip("per-channel input scale")
    return xq, int(exponent(sx)), pads


def match_bias(init, out_channels, channel_dim):
    op = producer(init)
    if op.name == "linalg.fill":
        if float(const(op.operands[0])) != 0.0:
            raise Skip("nonzero fill")
        return None, None
    op = producer(init, "linalg.broadcast")
    dims = i64_array(op, "dimensions")
    rank = len(shape(init))
    if dims != [d for d in range(rank) if d != channel_dim]:
        raise Skip("bias broadcast")
    bq, sb = dequant(op.operands[0], bits=32)
    b = const(bq).astype(np.int64)
    sb = np.broadcast_to(sb, (out_channels,))
    return b, exponent(sb)


def match(q):
    """A quantize_affine op -> description of the replacement, or Skip."""
    if len(q.operands) != 3 or attr_str(q, "quant_min") != "-128 : i64" \
            or attr_str(q, "quant_max") != "127 : i64":
        raise Skip("quantize range")
    sy = const(q.operands[1])
    if sy.ndim != 0:
        raise Skip("per-channel output scale")
    y_exp = int(exponent(sy))

    relu = False
    op = producer(q.operands[0])
    if is_relu(op):
        relu = True
        op = producer(op.operands[0])

    m = {"op": op.name.split(".")[-1], "relu": relu, "y_exp": y_exp, "y_type": q.results[0].type}
    if op.name in ("linalg.conv_2d_nchw_fchw", "linalg.depthwise_conv_2d_nchw_chw",
                   "linalg.conv_1d_ncw_fcw"):
        strides = [int(v) for v in np.array(op.attributes["strides"]).reshape(-1)]
        dilations = [int(v) for v in np.array(op.attributes["dilations"]).reshape(-1)]
        xq, x_exp, pads = match_input(op.operands[0])
        x_shape = shape(xq)
        if x_shape[0] != 1:
            raise Skip("batch != 1")
        wv = op.operands[1]
        if op.name == "linalg.depthwise_conv_2d_nchw_chw":
            wv = producer(wv, "tensor.collapse_shape").operands[0]
        wq, sw = dequant(wv)
        w = const(wq)  # (O, C/G, KH, KW) or (O, C/G, K)
        O = w.shape[0]
        if op.name == "linalg.conv_1d_ncw_fcw":
            C, H, W = x_shape[1], 1, x_shape[2]
            w = w[:, :, None, :]
            strides, dilations = [1, strides[0]], [1, dilations[0]]
            in_perm, out_perm, out_rank = [0, 2, 1], [0, 2, 1], 3
        else:
            C, H, W = x_shape[1], x_shape[2], x_shape[3]
            in_perm, out_perm, out_rank = [0, 2, 3, 1], [0, 3, 1, 2], 4
        G = C if op.name == "linalg.depthwise_conv_2d_nchw_chw" else 1
        if w.shape[1] * G != C:
            raise Skip("weight channels")
        w_ohwi = np.ascontiguousarray(w.transpose(0, 2, 3, 1))
        bias, b_exp = match_bias(op.operands[2], O, 1)
        m.update(xq=xq, x_exp=x_exp, in_perm=in_perm, out_perm=out_perm, out_rank=out_rank,
                 H=H, W=W, C=C, O=O, KH=w.shape[2], KW=w.shape[3],
                 SH=strides[0], SW=strides[1], DH=dilations[0], DW=dilations[1],
                 PT=pads[0], PL=pads[1], PB=pads[2], PR=pads[3], G=G,
                 w=w_ohwi, sw=np.broadcast_to(sw, (O,)), bias=bias, b_exp=b_exp)
    elif op.name == "linalg.batch_matmul":
        lhs = op.operands[0]
        lop = producer(lhs)
        if lop.name == "linalg.transpose":
            if i64_array(lop, "permutation") != [0, 2, 1]:
                raise Skip("matmul input permutation")
            xq, sx = dequant(lop.operands[0])  # (1, K, P): channels-first
            in_perm = [0, 2, 1]
        else:
            xq, sx = dequant(lhs)  # (1, P, K): channels-last
            in_perm = None
        if sx.ndim != 0 or shape(xq)[0] != 1:
            raise Skip("matmul input")
        bop = producer(op.operands[1], "linalg.generic")  # broadcast over batch
        body = [o.operation.name for o in bop.regions[0].blocks[0].operations]
        if body != ["linalg.yield"] or len(bop.operands) != 2:
            raise Skip("matmul weight broadcast")
        wq, sw = dequant(bop.operands[0])
        w = const(wq)  # (K, N)
        K, N = w.shape
        if sw.ndim == 1 and sw.shape[0] != N or sw.ndim > 1:
            raise Skip("matmul weight scale axis")
        bias, b_exp = match_bias(op.operands[2], N, 2)
        if bias is not None:
            raise Skip("matmul init")
        P = shape(lhs)[1]
        m.update(xq=xq, x_exp=int(exponent(sx)), in_perm=in_perm, out_perm=None, out_rank=3,
                 H=1, W=P, C=K, O=N, KH=1, KW=1, SH=1, SW=1, DH=1, DW=1,
                 PT=0, PL=0, PB=0, PR=0, G=1,
                 w=np.ascontiguousarray(w.T[:, None, None, :]),
                 sw=np.broadcast_to(sw, (N,)), bias=None, b_exp=None)
    else:
        raise Skip(f"producer {op.name}")

    m["OH"] = (m["H"] + m["PT"] + m["PB"] - m["DH"] * (m["KH"] - 1) - 1) // m["SH"] + 1
    m["OW"] = (m["W"] + m["PL"] + m["PR"] - m["DW"] * (m["KW"] - 1) - 1) // m["SW"] + 1
    w_exp = exponent(m["sw"])
    acc_exp = m["x_exp"] + w_exp
    if m["bias"] is None:
        bias = np.zeros(m["O"], dtype=np.int64)
    else:
        # Bias in the accumulator's scale: an integer only if its scale is
        # not finer than input * weight.
        if np.any(m["b_exp"] < acc_exp):
            raise Skip("bias scale finer than input * weight")
        bias = m["bias"] * (np.int64(1) << (m["b_exp"] - acc_exp))
    if np.any(np.abs(bias) > 2**31 - 1) or np.any(np.abs(w_exp) > 127):
        raise Skip("bias or exponent range")
    m["bias_acc"] = bias.astype(np.int32)
    m["w_exp"] = w_exp.astype(np.int8)
    m["x_shape"] = shape(m["xq"])
    m["X_EXP"], m["Y_EXP"], m["RELU"] = m["x_exp"], m["y_exp"], int(m["relu"])
    return m


# ---- Emission -------------------------------------------------------------------

EXECUTABLE = """
stream.executable private @espdl_executable {
  stream.executable.export public @conv2d_s8 workgroups() -> (index, index, index) {
    %c1 = arith.constant 1 : index
    hal.return %c1, %c1, %c1 : index, index, index
  }
  builtin.module {
    func.func private @espdl_conv2d_s8(
        %x: memref<i8>, %x_offset: index, %w: memref<i8>, %w_offset: index,
        %bias: memref<i32>, %bias_offset: index, %w_exp: memref<i8>, %w_exp_offset: index,
        %params: memref<i32>, %params_offset: index, %y: memref<i8>, %y_offset: index)
        attributes {llvm.bareptr}
    func.func @conv2d_s8(
        %x_b: !stream.binding, %w_b: !stream.binding, %bias_b: !stream.binding,
        %w_exp_b: !stream.binding, %params_b: !stream.binding, %y_b: !stream.binding,
        %nx: i32, %nw: i32, %no: i32, %np: i32, %ny: i32) {
      %c0 = arith.constant 0 : index
      %nxi = arith.index_castui %nx : i32 to index
      %nwi = arith.index_castui %nw : i32 to index
      %noi = arith.index_castui %no : i32 to index
      %npi = arith.index_castui %np : i32 to index
      %nyi = arith.index_castui %ny : i32 to index
      %x = stream.binding.subspan %x_b[%c0] : !stream.binding -> memref<?xi8>{%nxi}
      %w = stream.binding.subspan %w_b[%c0] : !stream.binding -> memref<?xi8>{%nwi}
      %bias = stream.binding.subspan %bias_b[%c0] : !stream.binding -> memref<?xi32>{%noi}
      %w_exp = stream.binding.subspan %w_exp_b[%c0] : !stream.binding -> memref<?xi8>{%noi}
      %params = stream.binding.subspan %params_b[%c0] : !stream.binding -> memref<?xi32>{%npi}
      %y = stream.binding.subspan %y_b[%c0] : !stream.binding -> memref<?xi8>{%nyi}
      %xp, %xo, %xs, %xt = iree_codegen.extract_strided_metadata %x : memref<?xi8> -> memref<i8>, index, index, index
      %wp, %wo, %ws, %wt = iree_codegen.extract_strided_metadata %w : memref<?xi8> -> memref<i8>, index, index, index
      %bp, %bo, %bs, %bt = iree_codegen.extract_strided_metadata %bias : memref<?xi32> -> memref<i32>, index, index, index
      %ep, %eo, %es, %et = iree_codegen.extract_strided_metadata %w_exp : memref<?xi8> -> memref<i8>, index, index, index
      %pp, %po, %ps, %pt = iree_codegen.extract_strided_metadata %params : memref<?xi32> -> memref<i32>, index, index, index
      %yp, %yo, %ys, %yt = iree_codegen.extract_strided_metadata %y : memref<?xi8> -> memref<i8>, index, index, index
      func.call @espdl_conv2d_s8(%xp, %xo, %wp, %wo, %bp, %bo, %ep, %eo, %pp, %po, %yp, %yo)
        : (memref<i8>, index, memref<i8>, index, memref<i32>, index, memref<i8>, index,
           memref<i32>, index, memref<i8>, index) -> ()
      return
    }
  }
}
"""


def tensor_t(dims, elem="i8"):
    return "tensor<" + "x".join(str(d) for d in dims) + f"x{elem}>"


def dense(arr, elem):
    data = np.ascontiguousarray(arr).astype({"i8": "<i1", "i32": "<i4"}[elem]).tobytes()
    return f'dense<"0x{data.hex().upper()}"> : tensor<{arr.size}x{elem}>'


def emit_func(name, m):
    x_shape = m["x_shape"]
    x_t = tensor_t(x_shape)
    nx = int(np.prod(x_shape))
    lines = [f"util.func private @{name}(%x: {x_t}) -> {m['y_type']} {{"]
    if m["in_perm"]:
        xt_shape = [x_shape[p] for p in m["in_perm"]]
        lines += [f"  %xe = tensor.empty() : {tensor_t(xt_shape)}",
                  f"  %xt = linalg.transpose ins(%x : {x_t}) outs(%xe : {tensor_t(xt_shape)}) "
                  f"permutation = [{', '.join(map(str, m['in_perm']))}]"]
        src, src_t = "%xt", tensor_t(xt_shape)
    else:
        src, src_t = "%x", x_t
    rank = len(x_shape)
    group = "[" + ", ".join(str(i) for i in range(rank)) + "]"
    O, OH, OW = m["O"], m["OH"], m["OW"]
    ny = OH * OW * O
    params = np.array([m[k] for k in PARAMS], dtype=np.int32)
    lines += [
        f"  %x1 = tensor.collapse_shape {src} [{group}] : {src_t} into tensor<{nx}xi8>",
        f"  %w = arith.constant {dense(m['w'], 'i8')}",
        f"  %b = arith.constant {dense(m['bias_acc'], 'i32')}",
        f"  %e = arith.constant {dense(m['w_exp'], 'i8')}",
        f"  %p = arith.constant {dense(params, 'i32')}",
        f"  %nx = arith.constant {nx} : i32",
        f"  %nw = arith.constant {m['w'].size} : i32",
        f"  %no = arith.constant {O} : i32",
        f"  %np = arith.constant {params.size} : i32",
        f"  %ny = arith.constant {ny} : i32",
        f"  %y1 = flow.dispatch @espdl_executable::@conv2d_s8(%x1, %w, %b, %e, %p, %nx, %nw, %no, %np, %ny)"
        f" : (tensor<{nx}xi8>, tensor<{m['w'].size}xi8>, tensor<{O}xi32>, tensor<{O}xi8>,"
        f" tensor<{params.size}xi32>, i32, i32, i32, i32, i32) -> tensor<{ny}xi8>",
    ]
    # Kernel output is channels-last: (1, OH, OW, O), or (1, OW, O) for rank 3.
    k_shape = [1, OH, OW, O] if m["out_rank"] == 4 else [1, OW, O]
    k_group = "[" + ", ".join(str(i) for i in range(len(k_shape))) + "]"
    lines.append(f"  %y2 = tensor.expand_shape %y1 [{k_group}] output_shape "
                 f"[{', '.join(map(str, k_shape))}] : tensor<{ny}xi8> into {tensor_t(k_shape)}")
    if m["out_perm"]:
        lines += [f"  %ye = tensor.empty() : {m['y_type']}",
                  f"  %y = linalg.transpose ins(%y2 : {tensor_t(k_shape)}) outs(%ye : {m['y_type']}) "
                  f"permutation = [{', '.join(map(str, m['out_perm']))}]",
                  f"  util.return %y : {m['y_type']}"]
    else:
        lines.append(f"  util.return %y2 : {m['y_type']}")
    lines.append("}")
    return "\n".join(lines)


def main():
    src, dst = sys.argv[1], sys.argv[2]
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.parse(open(src).read())
        matches, skipped = [], {}
        for op in walk(module.operation):
            if op.name != "iree_linalg_ext.quantize_affine":
                continue
            try:
                matches.append((op, match(op)))
            except Skip as e:
                skipped[str(e)] = skipped.get(str(e), 0) + 1

        calls = []
        untied = ir.ArrayAttr.get([ir.IntegerAttr.get(ir.IndexType.get(), -1)])
        for i, (q, m) in enumerate(matches):
            with ir.InsertionPoint(q):
                call = ir.Operation.create(
                    "util.call", results=[m["y_type"]], operands=[m["xq"]],
                    attributes={"callee": ir.FlatSymbolRefAttr.get(f"espdl_conv2d_s8_{i}"),
                                "tied_operands": untied})
            calls.append(call)
        # The input of one match may be the result of an earlier one: replace
        # only once all calls exist, so their operands get replaced too.
        for (q, _), call in zip(matches, calls):
            q.results[0].replace_all_uses_with(call.results[0])
            q.erase()

        funcs = [emit_func(f"espdl_conv2d_s8_{i}", m) for i, (_, m) in enumerate(matches)]
        if funcs:
            extra = ir.Module.parse("module {\n" + EXECUTABLE + "\n".join(funcs) + "\n}")
            body = module.body
            for op in list(extra.body.operations):
                op.operation.move_before(body.operations[0])
        with open(dst, "w") as f:
            f.write(str(module))
    kinds = {}
    for _, m in matches:
        k = f"{m['op']} {m['KH']}x{m['KW']}" + (f" s{m['SW']}" if m["SW"] > 1 else "") + (" relu" if m["relu"] else "")
        kinds[k] = kinds.get(k, 0) + 1
    print(f"espdl: {len(matches)} ops to espdl_conv2d_s8 {dict(sorted(kinds.items()))}")
    if skipped:
        print("left to IREE:", dict(sorted(skipped.items(), key=lambda kv: -kv[1])))


if __name__ == "__main__":
    main()
