"""Replace quantized ops of an imported esp-dl-quantized model with dispatches
to esp-dl kernels.

Runs on IREE's input-phase IR (iree-compile --compile-to=input). An imported
QDQ ONNX model is there float linalg ops between
iree_linalg_ext.dequantize_affine and quantize_affine. For each quantize_affine
that ends a quantized convolution or matrix product, this script reads the
int8 input, the int8 weights, int32 bias and scales, checks them against
esp-dl's scheme, and replaces the result with a dispatch of an esp-dl kernel
import. Kernels are channels-last; ops between them are rewritten to
channels-last too (see Layout), so transposes remain only at the edges.

Recognized, ending in quantize_affine (optionally after a ReLU):
  conv_2d_nchw_fchw, depthwise_conv_2d_nchw_chw, conv_1d_ncw_fcw
      input: dequantize_affine(x), optionally through tensor.pad with 0
      weights: dequantize_affine(w) (depthwise: then collapse_shape)
      init: broadcast(dequantize_affine(bias)) or fill 0
      -> espdl_conv2d_s8
  batch_matmul with batch 1 and constant weights
      input: dequantize_affine(x), optionally transposed [0, 2, 1]
      weights: broadcast of dequantize_affine(w) over the batch
      init: fill 0
      -> espdl_conv2d_s8
  elementwise ops on one dequantize_affine(x) with scalar constants
      -> espdl_lut_s8, a table evaluated with IREE on the host
  addf / mulf of two dequantize_affine, the second broadcast along trailing dims
      -> espdl_binary_s8
Anything else, including scales that are not powers of two, is left to IREE.

The imports are resolved at run time: host_plugin.c on the host, the esp-dl
adapter on ESP32-P4. Their contract is espdl_ref.h.

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
    return (attr_str(cmp, "predicate") in ("2 : i64", "9 : i64")  # ogt, ugt
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

    m = {"kind": "conv", "op": op.name.split(".")[-1], "relu": relu, "y_exp": y_exp, "y_type": q.results[0].type}
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


# ---- Unary and binary elementwise ----------------------------------------------

def scalar_exp(op):
    """Exponent of a quantize/dequantize_affine with one power-of-two scale."""
    if len(op.operands) != 3 or attr_str(op, "quant_min") != "-128 : i64" \
            or attr_str(op, "quant_max") != "127 : i64":
        raise Skip("quantize range or zero point")
    s = const(op.operands[1])
    if s.ndim != 0:
        raise Skip("per-channel scale")
    return int(exponent(s))


def is_elementwise(op):
    if op.name != "linalg.generic" or len(op.results) != 1:
        return False
    if not all("parallel" in str(t) for t in ir.ArrayAttr(op.attributes["iterator_types"])):
        return False
    return not any(o.name == "linalg.index" for o in walk(op))


def map_kind(attr):
    """'id' for an identity map, 'scalar' for (...) -> (), else None."""
    am = ir.AffineMapAttr(attr).value
    if len(am.results) == 0:
        return "scalar"
    if len(am.results) == am.n_dims and all(
            str(r) == f"d{i}" for i, r in enumerate(am.results)):
        return "id"
    return None


def captured(op):
    """Top-level arith.constant values that op's body reads."""
    out = []
    for o in walk(op):
        for v in o.operands:
            owner = v.owner
            if not isinstance(owner, ir.Block) and owner.operation.name == "arith.constant" \
                    and owner.operation.parent.name == "util.func":
                out.append(v)
    return out


def match_unary(q):
    """int8 x -> elementwise ops -> quantize: a 256-entry table. Returns the
    chain (producers first) for evaluating the table."""
    scalar_exp(q)
    chain, leaves = [], set()

    def visit(v):
        op = producer(v)
        if op in chain:
            return
        if op.name == "arith.constant":
            if shape(v) and not ir.DenseElementsAttr(op.attributes["value"]).is_splat:
                raise Skip("non-splat constant")
            return
        if op.name == "iree_linalg_ext.dequantize_affine":
            scalar_exp(op)
            leaves.add(op.operands[0])
        elif op.name == "iree_linalg_ext.quantize_affine":
            scalar_exp(op)
            visit(op.operands[0])
        elif is_elementwise(op):
            maps = list(ir.ArrayAttr(op.attributes["indexing_maps"]))
            if any(map_kind(a) is None for a in maps) or \
                    producer(op.operands[-1]).name != "tensor.empty":
                raise Skip("elementwise with broadcast or non-empty init")
            for x in list(op.operands)[:-1]:
                visit(x)
        else:
            raise Skip(f"producer {op.name}")
        chain.append(op)

    visit(q.operands[0])
    if len(leaves) != 1:
        raise Skip(f"{len(leaves)} quantized inputs")
    x = next(iter(leaves))
    if str(ir.RankedTensorType(x.type).element_type) != "i8" or shape(x) != shape(q.results[0]):
        raise Skip("unary shape")
    return {"kind": "lut", "op": "lut", "xq": x, "chain": chain + [q], "y_type": q.results[0].type}


def match_binary(q):
    """dequantize(a) (+|*) dequantize(b) -> quantize; b may broadcast."""
    ey = scalar_exp(q)
    op = producer(q.operands[0])
    if not is_elementwise(op) or len(op.operands) != 3:
        raise Skip("not binary")
    body = [o.operation.name for o in op.regions[0].blocks[0].operations]
    if body not in (["arith.addf", "linalg.yield"], ["arith.mulf", "linalg.yield"]):
        raise Skip("binary body")
    if producer(op.operands[2]).name != "tensor.empty":
        raise Skip("binary init")
    maps = list(ir.ArrayAttr(op.attributes["indexing_maps"]))
    sides = []
    for v, mp in zip(list(op.operands)[:2], maps[:2]):
        dq = producer(v, "iree_linalg_ext.dequantize_affine")
        sides.append((dq.operands[0], scalar_exp(dq), mp))
    rank = len(shape(op.results[0]))
    if map_kind(sides[0][2]) != "id":
        sides.reverse()  # both ops are commutative
    (a, ea, ma), (b, eb, mb) = sides
    if map_kind(ma) != "id":
        raise Skip("binary: neither side full")
    # b's shape aligned to the output's dims (1 where broadcast).
    b_full = [1] * rank
    bam = ir.AffineMapAttr(mb).value
    for k, r in enumerate(bam.results):
        if str(r).startswith("d"):
            b_full[int(str(r)[1:])] = shape(b)[k]
        elif str(r) != "0" or shape(b)[k] != 1:
            raise Skip("binary broadcast map")
    kind = "add" if body[0] == "arith.addf" else "mul"
    if kind == "add" and abs(ea - eb) > 15:
        raise Skip("add exponents too far apart for exact float")
    for v in (a, b):
        if str(ir.RankedTensorType(v.type).element_type) != "i8":
            raise Skip("binary input not int8")
    return {"kind": "binary", "op": kind, "a": a, "b": b, "ea": ea, "eb": eb, "ey": ey,
            "a_shape": shape(a), "b_full": b_full, "y_type": q.results[0].type}


def evaluate_luts(luts):
    """Run each unary chain on -128..127 with IREE on the host: the tables
    hold exactly what IREE computed for these ops before."""
    from iree import compiler as ireec
    from iree import runtime as rt

    iota = np.arange(-128, 128, dtype=np.int8)
    parallel = ir.Attribute.parse("[#linalg.iterator_type<parallel>]")
    id1 = ir.AffineMapAttr.get(ir.AffineMap.get(1, 0, [ir.AffineDimExpr.get(0)]))
    sc1 = ir.AffineMapAttr.get(ir.AffineMap.get(1, 0, []))
    funcs = []
    for i, m in enumerate(luts):
        funcs.append(f"func.func @lut_{i}(%x: tensor<256xi8>) -> tensor<256xi8> {{\n"
                     f"  return %x : tensor<256xi8>\n}}")
    mod = ir.Module.parse("\n".join(funcs))
    for i, m in enumerate(luts):
        func = mod.body.operations[i]
        block = func.regions[0].blocks[0]
        ret = list(block.operations)[-1]
        mapping = {m["xq"]: block.arguments[0]}

        def remap(v):
            if v in mapping:
                return mapping[v]
            owner = v.owner.operation
            if owner.name == "arith.constant":
                with ir.InsertionPoint(ret):
                    mapping[v] = owner.clone().results[0]
                return mapping[v]
            raise Skip("unexpected chain operand")

        with ir.InsertionPoint(ret):
            for op in m["chain"]:
                for v in captured(op):
                    remap(v)
                c = op.clone()
                n = len(c.operands)
                elem = ir.RankedTensorType(op.results[0].type).element_type
                t256 = ir.RankedTensorType.get([256], elem)
                for k in range(n - 1):
                    c.operands[k] = remap(op.operands[k])
                c.operands[n - 1] = ir.Operation.create("tensor.empty", results=[t256], operands=[]).results[0]
                c.move_after(c.operands[n - 1].owner.operation)
                for o in walk(c):
                    for k, v in enumerate(o.operands):
                        if v in mapping:
                            o.operands[k] = mapping[v]
                c.results[0].set_type(t256)
                c.attributes["indexing_maps"] = ir.ArrayAttr.get(
                    [id1 if map_kind(a) == "id" else sc1 for a in ir.ArrayAttr(op.attributes["indexing_maps"])])
                if "iterator_types" in c.attributes:
                    c.attributes["iterator_types"] = parallel
                mapping[op.results[0]] = c.results[0]
        ret.operands[0] = mapping[m["chain"][-1].results[0]]
    vmfb = ireec.compile_str(str(mod), target_backends=["llvm-cpu"],
                             extra_args=["--iree-llvmcpu-target-cpu=generic"])
    config = rt.Config("local-sync")
    ctx = rt.SystemContext(config=config)
    ctx.add_vm_module(rt.VmModule.copy_buffer(ctx.instance, vmfb))
    module = ctx.modules.module
    for i, m in enumerate(luts):
        m["table"] = np.asarray(module[f"lut_{i}"](iota).to_host()).astype(np.int8)


# ---- Emission -------------------------------------------------------------------

def pack_filter(w, depthwise):
    """OHWI int8 weights in esp-dl's ESP32-P4 layout (ESP-PPQ's
    ResetParamLayoutPattern): output channels in blocks of 16, each block
    (H, W, I, 16); the remaining channels after them as OHWI, or for depthwise
    (H, W, O)."""
    o, kh, kw, c = w.shape
    al = o // 16 * 16
    head = w[:al].reshape(o // 16, 16, kh, kw, c).transpose(0, 2, 3, 4, 1)
    tail = w[al:]
    if depthwise:
        tail = tail.transpose(3, 1, 2, 0)  # (1, H, W, O_rem)
    return np.concatenate([head.reshape(-1), tail.reshape(-1)])


# Imports: bindings in order, the last one the output. Each is passed to the
# import as a base pointer and an element offset (espdl_ref.h, host_plugin.c).
IMPORTS = {
    "conv2d_s8": [("x", "i8"), ("w", "i8"), ("bias", "i32"), ("w_exp", "i8"),
                  ("params", "i32"), ("y", "i8")],
    "lut_s8": [("x", "i8"), ("table", "i8"), ("params", "i32"), ("y", "i8")],
    "binary_s8": [("a", "i8"), ("b", "i8"), ("params", "i32"), ("y", "i8")],
}


def executable_text(used):
    exports, funcs = [], []
    for name in used:
        binds = IMPORTS[name]
        exports.append(
            f"  stream.executable.export public @{name} workgroups() -> (index, index, index) {{\n"
            f"    %c1 = arith.constant 1 : index\n"
            f"    hal.return %c1, %c1, %c1 : index, index, index\n  }}")
        decl = ", ".join(f"memref<{t}>, index" for _, t in binds)
        lines = [f"    func.func private @espdl_{name}({decl}) attributes {{llvm.bareptr}}",
                 f"    func.func @{name}(" +
                 ", ".join(f"%{n}_b: !stream.binding" for n, _ in binds) + ", " +
                 ", ".join(f"%{n}_n: i32" for n, _ in binds) + ") {",
                 "      %c0 = arith.constant 0 : index"]
        args, types = [], []
        for n, t in binds:
            lines += [
                f"      %{n}_ni = arith.index_castui %{n}_n : i32 to index",
                f"      %{n} = stream.binding.subspan %{n}_b[%c0] : !stream.binding -> memref<?x{t}>{{%{n}_ni}}",
                f"      %{n}_p, %{n}_o, %{n}_s, %{n}_t = iree_codegen.extract_strided_metadata %{n}"
                f" : memref<?x{t}> -> memref<{t}>, index, index, index"]
            args += [f"%{n}_p", f"%{n}_o"]
            types += [f"memref<{t}>", "index"]
        lines += [f"      func.call @espdl_{name}({', '.join(args)}) : ({', '.join(types)}) -> ()",
                  "      return", "    }"]
        funcs.append("\n".join(lines))
    return ("stream.executable private @espdl_executable {\n" + "\n".join(exports) +
            "\n  builtin.module {\n" + "\n".join(funcs) + "\n  }\n}\n")


def tensor_t(dims, elem="i8"):
    return "tensor<" + "x".join(str(d) for d in dims) + f"x{elem}>"


def dense(arr, elem):
    data = np.ascontiguousarray(arr).astype({"i8": "<i1", "i32": "<i4"}[elem]).tobytes()
    return f'dense<"0x{data.hex().upper()}"> : tensor<{arr.size}x{elem}>'


def emit_dispatch(fname, imp, args, consts, out_shape):
    """util.func @fname(args...) dispatching import imp. args: shapes of the
    function's int8 arguments; consts: (array, elem) bound after them."""
    binds = IMPORTS[imp]
    arg_ts = [tensor_t(a) for a in args]
    lines = [f"util.func private @{fname}(" +
             ", ".join(f"%a{i}: {t}" for i, t in enumerate(arg_ts)) + f") -> {tensor_t(out_shape)} {{"]
    vals, types, sizes = [f"%a{i}" for i in range(len(args))], list(arg_ts), \
        [int(np.prod(a)) for a in args]
    for k, (arr, elem) in enumerate(consts):
        lines.append(f"  %k{k} = arith.constant {dense(arr, elem)}")
        vals.append(f"%k{k}")
        types.append(f"tensor<{arr.size}x{elem}>")
        sizes.append(int(arr.size))
    sizes.append(int(np.prod(out_shape)))
    assert len(sizes) == len(binds)
    for k, n in enumerate(sizes):
        lines.append(f"  %n{k} = arith.constant {n} : i32")
    lines += [f"  %y = flow.dispatch @espdl_executable::@{imp}({', '.join(vals)}, " +
              ", ".join(f"%n{k}" for k in range(len(sizes))) + ") : (" +
              ", ".join(types) + ", " + ", ".join("i32" for _ in sizes) +
              f") -> {tensor_t(out_shape)}",
              f"  util.return %y : {tensor_t(out_shape)}", "}"]
    return "\n".join(lines)


def emit_func(name, m):
    """The kernel call for one match."""
    if m["kind"] == "conv":
        params = np.array([m[k] for k in PARAMS], dtype=np.int32)
        return emit_dispatch(name, "conv2d_s8", [m["k_in_shape"]],
                             [(pack_filter(m["w"], m["G"] > 1), "i8"), (m["bias_acc"], "i32"),
                              (m["w_exp"], "i8"),
                              (params, "i32")], m["k_shape"])
    if m["kind"] == "lut":
        n = int(np.prod(m["k_shape"]))
        return emit_dispatch(name, "lut_s8", [m["k_shape"]],
                             [(m["table"], "i8"), (np.array([n], np.int32), "i32")], m["k_shape"])
    params = np.array(m["params"], dtype=np.int32)
    return emit_dispatch(name, "binary_s8", [m["k_a_shape"], m["k_b_shape"]],
                         [(params, "i32")], m["k_shape"])


# ---- Layout ---------------------------------------------------------------------
#
# The kernels are channels-last; the imported model is channels-first. Rather
# than transposing around every kernel, the channels-last result of a kernel is
# carried forward: an op that reads it (elementwise, reduction, quantize,
# dequantize) is cloned to read the channels-last value instead, by permuting
# that operand's indexing map, and to write channels-last too when its output
# has the same rank. Iteration spaces do not change, so the ops compute the
# same values. Ops that need the channels-first value get one transpose.

# Channels-last permutation by rank: nhwc[j] = nchw[PERM[j]].
PERM = {4: [0, 2, 3, 1], 3: [0, 2, 1]}
LAYOUT_OPS = ("linalg.generic", "iree_linalg_ext.quantize_affine",
              "iree_linalg_ext.dequantize_affine", "tensor.concat",
              "linalg.pooling_nchw_sum", "linalg.pooling_nchw_max", "tensor.pad")


def permuted_type(t, perm):
    t = ir.RankedTensorType(t)
    return ir.RankedTensorType.get([t.shape[p] for p in perm], t.element_type)


def permuted_map(attr, perm):
    am = ir.AffineMapAttr(attr).value
    return ir.AffineMapAttr.get(
        ir.AffineMap.get(am.n_dims, am.n_symbols, [am.results[p] for p in perm]))


class Layout:
    def __init__(self, untied):
        self.untied = untied
        self.nhwc = {}        # channels-first value -> its channels-last version
        self.pending = []     # (old op, channels-last result or None, replacement or None)
        self.helpers = {}     # helper name -> text
        self.helpers_by_key = {}

    def call(self, name, values, result_type):
        values = values if isinstance(values, list) else [values]
        return ir.Operation.create(
            "util.call", results=[result_type], operands=values,
            attributes={"callee": ir.FlatSymbolRefAttr.get(name),
                        "tied_operands": self.untied}).results[0]

    def transpose(self, value, perm, prefix):
        t = ir.RankedTensorType(value.type)
        out = permuted_type(t, perm)
        name = f"espdl_{prefix}_" + "x".join(map(str, t.shape)) + f"x{t.element_type}"
        if name not in self.helpers:
            src_t, dst_t = str(t), str(out)
            self.helpers[name] = "\n".join([
                f"util.func private @{name}(%x: {src_t}) -> {dst_t} {{",
                f"  %e = tensor.empty() : {dst_t}",
                f"  %y = linalg.transpose ins(%x : {src_t}) outs(%e : {dst_t}) "
                f"permutation = [{', '.join(map(str, perm))}]",
                f"  util.return %y : {dst_t}",
                "}"])
        return self.call(name, value, out)

    def to_nhwc(self, value):
        """Channels-last version of a channels-first value (inserts a transpose)."""
        if value not in self.nhwc:
            self.nhwc[value] = self.transpose(value, PERM[len(shape(value))], "to_nhwc")
        return self.nhwc[value]

    def kernel(self, q, m, name):
        """Replace quantize q by kernel call name. Returns False if the
        operands' layouts do not fit the kernel."""
        i8 = ir.IntegerType.get_signless(8)
        if m["kind"] == "conv":
            x = m["xq"]
            kin = self.to_nhwc(x) if m["in_perm"] else x
            y = self.call(name, [kin], ir.RankedTensorType.get(m["k_shape"], i8))
            if m["out_perm"]:
                self.nhwc[q.results[0]] = y
                self.pending.append((q, y, None))
            else:
                self.pending.append((q, None, y))
            return True
        if m["kind"] == "lut":
            x = m["xq"]
            kin = self.nhwc.get(x, x)
            m["k_shape"] = shape(kin)
            y = self.call(name, [kin], ir.RankedTensorType.get(m["k_shape"], i8))
            self.record(q, y, x in self.nhwc)
            return True
        # binary: pick the layout, then check that b repeats along a suffix.
        a, b = m["a"], m["b"]
        rank = len(m["a_shape"])
        cands = []
        if rank in PERM and (a in self.nhwc or b in self.nhwc):
            cands.append(PERM[rank])
        cands.append(list(range(rank)))
        if rank in PERM and PERM[rank] not in cands:
            cands.append(PERM[rank])
        for perm in cands:
            ap = [m["a_shape"][p] for p in perm]
            bp = [m["b_full"][p] for p in perm]
            k = 0
            while k < rank and bp[k] == 1:
                k += 1
            if bp[k:] != ap[k:]:
                continue
            nhwc = perm != list(range(rank))
            av = (self.to_nhwc(a) if nhwc else a)
            bv = b
            if len(shape(b)) == rank and nhwc:
                bv = self.to_nhwc(b)
            m["k_a_shape"], m["k_b_shape"] = shape(av), shape(bv)
            m["k_shape"] = ap
            n, bl = int(np.prod(ap)), int(np.prod(bp))
            m["params"] = [n, bl, {"add": 0, "mul": 1}[m["op"]], m["ea"], m["eb"], m["ey"]]
            y = self.call(name, [av, bv], ir.RankedTensorType.get(ap, i8))
            self.record(q, y, nhwc)
            return True
        return False

    def record(self, q, y, nhwc):
        if nhwc:
            self.nhwc[q.results[0]] = y
            self.pending.append((q, y, None))
        else:
            self.pending.append((q, None, y))

    def init_like(self, init, out_type):
        """A channels-last init for a DPS op: tensor.empty, or fill of one."""
        op = producer(init)
        if op.name == "tensor.empty" and not len(op.operands):
            return ir.Operation.create("tensor.empty", results=[out_type], operands=[]).results[0]
        if op.name == "linalg.fill":
            empty = self.init_like(op.operands[1], out_type)
            if empty is None:
                return None
            fill = op.clone()
            fill.operands[1] = empty
            fill.results[0].set_type(out_type)
            return fill.results[0]
        return None

    def convert(self, op):
        """Clone a layout-agnostic op to read (and write) channels-last values."""
        if op.name == "tensor.concat":
            return self.convert_concat(op)
        if op.name.startswith("linalg.pooling_nchw_"):
            return self.convert_pooling(op)
        if op.name == "tensor.pad":
            return self.convert_pad(op)
        if len(op.results) != 1 or "indexing_maps" not in op.attributes:
            return
        operands = list(op.operands)
        maps = list(ir.ArrayAttr(op.attributes["indexing_maps"]))
        if len(maps) != len(operands):
            return
        ins = [i for i, v in enumerate(operands[:-1]) if v in self.nhwc]
        # Values the body reads with tensor.extract (Resize).
        extracts = [o for o in walk(op) if o.name == "tensor.extract" and o.operands[0] in self.nhwc]
        if not ins and not extracts:
            return
        out_rank = len(shape(op.results[0]))
        new_init = None
        read_ranks = [len(shape(operands[i])) for i in ins] + [len(shape(o.operands[0])) for o in extracts]
        if out_rank in PERM and out_rank in read_ranks:
            new_init = self.init_like(operands[-1], permuted_type(op.results[0].type, PERM[out_rank]))
        clone = op.clone()
        for i in ins:
            clone.operands[i] = self.nhwc[operands[i]]
            maps[i] = permuted_map(maps[i], PERM[len(shape(operands[i]))])
        for o in walk(clone):
            if o.name == "tensor.extract" and o.operands[0] in self.nhwc:
                perm = PERM[len(shape(o.operands[0]))]
                idx = list(o.operands)[1:]
                o.operands[0] = self.nhwc[o.operands[0]]
                for j, p in enumerate(perm):
                    o.operands[1 + j] = idx[p]
        if new_init is not None:
            clone.operands[len(operands) - 1] = new_init
            maps[-1] = permuted_map(maps[-1], PERM[out_rank])
            clone.results[0].set_type(new_init.type)
        clone.attributes["indexing_maps"] = ir.ArrayAttr.get(maps)
        if new_init is not None:
            self.nhwc[op.results[0]] = clone.results[0]
            self.pending.append((op, clone.results[0], None))
        else:
            self.pending.append((op, None, clone.results[0]))

    def convert_concat(self, op):
        operands = list(op.operands)
        rank = len(shape(op.results[0]))
        if rank not in PERM or not any(v in self.nhwc for v in operands):
            return
        perm = PERM[rank]
        dim = int(ir.IntegerAttr(op.attributes["dim"]).value)
        inputs = [self.to_nhwc(v) for v in operands]
        clone = op.clone()
        for i, v in enumerate(inputs):
            clone.operands[i] = v
        clone.attributes["dim"] = ir.IntegerAttr.get(ir.IntegerType.get_signless(64), perm.index(dim))
        clone.results[0].set_type(permuted_type(op.results[0].type, perm))
        self.nhwc[op.results[0]] = clone.results[0]
        self.pending.append((op, clone.results[0], None))

    def convert_pad(self, op):
        x = op.operands[0]
        if x not in self.nhwc or len(op.operands) != 1:  # static padding only
            return
        perm = PERM[len(shape(x))]
        clone = op.clone()
        clone.operands[0] = self.nhwc[x]
        for name in ("static_low", "static_high"):
            pads = i64_array(op, name)
            clone.attributes[name] = ir.DenseI64ArrayAttr.get([pads[p] for p in perm])
        clone.results[0].set_type(permuted_type(op.results[0].type, perm))
        self.nhwc[op.results[0]] = clone.results[0]
        self.pending.append((op, clone.results[0], None))

    def convert_pooling(self, op):
        x, window, init = op.operands
        if x not in self.nhwc:
            return
        fill = producer(init)
        if fill.name != "linalg.fill" or producer(fill.operands[0]).name != "arith.constant":
            return
        value = producer(fill.operands[0]).attributes["value"]
        kind = op.name.split("_")[-1]
        xt, wt = str(self.nhwc[x].type), str(window.type)
        yt = str(permuted_type(op.results[0].type, PERM[4]))
        attrs = f"dilations = {op.attributes['dilations']}, strides = {op.attributes['strides']}"
        key = f"{kind} {xt} {wt} {attrs} {value}"
        name = self.helpers_by_key.get(key)
        if name is None:
            name = f"espdl_pool_nhwc_{len(self.helpers_by_key)}"
            self.helpers_by_key[key] = name
            self.helpers[name] = "\n".join([
                f"util.func private @{name}(%x: {xt}) -> {yt} {{",
                f"  %c = arith.constant {value}",
                f"  %e = tensor.empty() : {yt}",
                f"  %init = linalg.fill ins(%c : {value.type}) outs(%e : {yt}) -> {yt}",
                f"  %w = tensor.empty() : {wt}",
                f"  %y = linalg.pooling_nhwc_{kind} {{{attrs}}} ins(%x, %w : {xt}, {wt}) "
                f"outs(%init : {yt}) -> {yt}",
                f"  util.return %y : {yt}",
                "}"])
        y = self.call(name, self.nhwc[x], ir.Type.parse(yt))
        self.nhwc[op.results[0]] = y
        self.pending.append((op, y, None))

    def finish(self):
        """Erase the replaced ops; channels-first users get a transpose."""
        for old, nhwc, repl in reversed(self.pending):
            result = old.results[0]
            if any(True for _ in result.uses):
                if repl is None:
                    with ir.InsertionPoint(old):
                        perm = PERM[len(shape(result))]
                        repl = self.transpose(nhwc, [perm.index(i) for i in range(len(perm))], "to_nchw")
                result.replace_all_uses_with(repl)
            old.erase()


PURE_DIALECTS = ("linalg.", "tensor.", "iree_linalg_ext.", "arith.")


def dce(module, keep=()):
    """Erase side-effect-free ops whose results are unused (the replaced
    convolutions and their inputs), except those in keep."""
    keep = set(keep)
    changed = True
    while changed:
        changed = False
        for func in module.body.operations:
            func = func.operation
            if func.name != "util.func" or not len(func.regions[0].blocks):
                continue
            for op in reversed(list(func.regions[0].blocks[0].operations)):
                op = op.operation
                pure = op.name.startswith(PURE_DIALECTS) or (
                    op.name == "util.call" and str(op.attributes["callee"]).startswith(("@espdl_to_", "@espdl_pool_")))
                if pure and op not in keep and len(op.results) and not any(True for r in op.results for _ in r.uses):
                    op.erase()
                    changed = True


def main():
    src, dst = sys.argv[1], sys.argv[2]
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.parse(open(src).read())
        matches, skipped = [], {}
        for op in walk(module.operation):
            if op.name != "iree_linalg_ext.quantize_affine":
                continue
            reasons = []
            for matcher in (match, match_unary, match_binary):
                try:
                    matches.append((op, matcher(op)))
                    break
                except Skip as e:
                    reasons.append(str(e))
            else:
                skipped[reasons[0]] = skipped.get(reasons[0], 0) + 1
        for _, m in matches:
            if m["kind"] == "conv":
                m["k_in_shape"] = [m["x_shape"][p] for p in m["in_perm"]] if m["in_perm"] else m["x_shape"]
                m["k_shape"] = [1, m["OH"], m["OW"], m["O"]] if m["out_rank"] == 4 else [1, m["OW"], m["O"]]
        evaluate_luts([m for _, m in matches if m["kind"] == "lut"])

        names = [f"espdl_{m['kind']}_{i}" for i, (_, m) in enumerate(matches)]
        index = {q: i for i, (q, _) in enumerate(matches)}
        layout = Layout(ir.ArrayAttr.get([ir.IntegerAttr.get(ir.IndexType.get(), -1)]))
        declined = set()
        for func in list(module.body.operations):
            func = func.operation
            if func.name != "util.func" or not len(func.regions[0].blocks):
                continue
            for op in list(func.regions[0].blocks[0].operations):
                op = op.operation
                with ir.InsertionPoint(op):
                    if op in index and layout.kernel(op, matches[index[op]][1], names[index[op]]):
                        continue
                    if op in index:
                        declined.add(index[op])
                    if op.name in LAYOUT_OPS:
                        layout.convert(op)
        dce(module, keep=[o for old, *new in layout.pending
                          for o in [old] + [v.owner.operation for v in new if v is not None]])
        layout.finish()
        dce(module)

        calls = {}
        for op in walk(module.operation):
            if op.name == "util.call":
                callee = str(op.attributes["callee"])[1:]
                calls[callee] = calls.get(callee, 0) + 1
        live = [(name, m) for i, (name, (_, m)) in enumerate(zip(names, matches)) if i not in declined]
        used = sorted({ {"conv": "conv2d_s8", "lut": "lut_s8", "binary": "binary_s8"}[m["kind"]]
                        for _, m in live})
        funcs = [emit_func(name, m) for name, m in live]
        funcs += [text for name, text in layout.helpers.items() if name in calls]
        if funcs:
            extra = ir.Module.parse("module {\n" + executable_text(used) + "\n".join(funcs) + "\n}")
            for op in list(extra.body.operations):
                op.operation.move_before(module.body.operations[0])
        with open(dst, "w") as f:
            f.write(str(module))
    kinds = {}
    for _, m in live:
        if m["kind"] == "conv":
            k = f"{m['op']} {m['KH']}x{m['KW']}" + (f" s{m['SW']}" if m["SW"] > 1 else "") + \
                (" relu" if m["relu"] else "")
        else:
            k = m["op"]
        kinds[k] = kinds.get(k, 0) + 1
    print(f"espdl: {len(live)} ops to esp-dl kernels {dict(sorted(kinds.items()))}")
    if skipped or declined:
        if declined:
            skipped["binary layout"] = len(declined)
        print("left to IREE:", dict(sorted(skipped.items(), key=lambda kv: -kv[1])))
    to_nhwc = sum(n for k, n in calls.items() if k.startswith("espdl_to_nhwc"))
    to_nchw = sum(n for k, n in calls.items() if k.startswith("espdl_to_nchw"))
    print(f"layout: channels-last between kernels; transposes left: {to_nhwc} to NHWC, {to_nchw} to NCHW")

if __name__ == "__main__":
    main()
