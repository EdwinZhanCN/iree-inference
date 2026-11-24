"""Route GEMMs and attention to MLX steel kernels on IREE Metal.

The kernels are kernels/mlx_gemm.metal and kernels/mlx_attention.metal,
linked into one metallib.

Writes a preprocessing transform spec (--iree-preprocessing-transform-spec-
filename) that replaces each `fill(0.0 : f32)` + `linalg.batch_matmul` of
f16 operands with the given static shapes by a hal.dispatch.extern call into
the metallib. The kernel writes f16 after f32 accumulation, which is what
IREE's own matmul + truncf computes; the result is extended back to f32 so the
original consumers (truncf + bias/residual epilogue) stay unchanged and get
fused by IREE.

External Metal variants need the patched MetalSPIRV target.

A --linear rule also takes in the transpose and batch-1 broadcast of a
torch linear's weight and uses the nt kernel on the original weight.

An --attention rule replaces torch's multi-head attention as it reaches
preprocessing: q, k and v [B, L, H*D] split into heads and transposed to
[B*H, L, D], iree_linalg_ext.attention with scale D^-0.5, and the transpose
back. MLX's kernel reads and writes [B, L, H, D] through strides, so none of
the transposes remain.

  mlx_kernels.py <mlx_kernels.metallib> <spec.mlir> [--linear MxNxK]...
      [--gemm BxMxNxK]... [--attention BxLxHxD]...
"""
import argparse
import struct
import subprocess
from pathlib import Path

# The default Metal executable target of iree-compile 3.13; objects are keyed
# by it, so it must match the device target the module compiles for.
METAL_TARGET = (
    '#hal.executable.target<"metal-spirv", "metal-msl-fb", {iree_codegen.target_info = '
    '#iree_gpu.target<arch = "apple", features = "spirv:v1.3,cap:Shader", wgp = <compute = '
    "fp32|fp16|int64|int32|int16|int8, storage = b64|b32|b16|b8, subgroup = shuffle|arithmetic, "
    "dot = dp4xi8toi32, subgroup_size_choices = [32], max_workgroup_sizes = [1024, 1024, 1024], "
    "max_thread_count_per_workgroup = 1024, max_workgroup_memory_bytes = 32768, "
    "max_workgroup_counts = [65535, 65535, 65535]>>}>"
)
ALIGNMENTS = {"aaa", "uaa", "aua", "uua"}  # instantiated, plus "uuu"


def detect_arch():
    """The local GPU's Metal architecture name, e.g. applegpu_g14s."""
    out = subprocess.run(["xcrun", "swift", "-e",
                          "import Metal; print(MTLCreateSystemDefaultDevice()!.architecture.name)"],
                         check=True, text=True, capture_output=True).stdout
    return out.strip().splitlines()[-1]


# MLX picks tiles at run time from the last letter of the architecture name
# (its device class: g and p small, d large, the rest medium). IREE compiles
# ahead of time, so the spec is written for one architecture: this machine's
# by default, or the one given for a cross-compile.
DEVICE_CLASS = None  # set with set_arch()
# Whether GEMMs and attention use MLX's NAX (Metal 4 tensor ops) kernels, as MLX's
# is_nax_available() decides: architecture generation 18 on phones (A19), 17
# elsewhere (M5). Those kernels are built only into the iOS metallib.
NAX = False


def set_arch(arch, nax=None):
    """nax=None follows MLX; True or False overrides it."""
    global DEVICE_CLASS, NAX
    DEVICE_CLASS = arch[-1]
    digits = arch[-3:-1]
    gen = int(digits) if digits.isdigit() else 0
    NAX = gen >= (18 if DEVICE_CLASS == "p" else 17) if nax is None else nax


def gemm_tile(batch, m, n, k, nt):
    """(BM, BN, BK, WM, WN) for an f16 GEMM, as MLX's GEMM_TPARAM_MACRO
    (mlx/backend/metal/matmul.cpp) chooses it. All of them are instantiated
    in mlx_gemm.metal."""
    devc = DEVICE_CLASS
    assert devc, "call set_arch() first"
    if devc in "gp":
        return (64, 32, 32, 2, 2) if nt else (64, 64, 16, 1, 2)
    if devc == "d":
        if batch * m * n >= 1 << 20:
            if 2 * max(m, n) > k:
                return (64, 64, 16, 1, 2)
            return (64, 32, 32, 2, 2) if nt else (32, 64, 16, 1, 2)
        return (64, 32, 32, 2, 2) if nt else (64, 64, 16, 1, 2)
    return (64, 64, 16, 2, 2)


# MLX's regular (non-NAX) steel GEMM dispatch uses no threadgroup swizzle.
SWIZZLE = 0


def gemm_nax_tile(m, n, k):
    """(BM, BN, BK, WM, WN) as MLX's steel_matmul_regular_axpby_nax picks it;
    all instantiated in mlx_gemm_nax.metal."""
    if DEVICE_CLASS in "sc":
        return (64, 128, 64 if k >= 8192 and k > m + n else 256, 2, 4)
    if DEVICE_CLASS == "d":
        return (128, 64, 512, 4, 2) if n > m else (64, 128, 512, 2, 4)
    return (128, 128, 512, 4, 4)


def gemm_nax_swizzle(m, n, bm):
    """MLX's swizzle for the NAX GEMM."""
    if DEVICE_CLASS in "sc":
        return 2
    if DEVICE_CLASS == "d":
        return 0 if m > n else 2
    return 0 if -(-m // bm) <= 3 else 1


def kernel_name(batch, m, n, k, nt=False):
    layout = "nt" if nt else "nn"
    if NAX:
        bm, bn, bk, wm, wn = gemm_nax_tile(m, n, k)
        align = "".join("a" if d % t == 0 else "u" for d, t in ((m, bm), (n, bn), (k, bk)))
        return f"mlx_gemm_nax_{layout}_f16_bm{bm}_bn{bn}_bk{bk}_wm{wm}_wn{wn}_{align}"
    bm, bn, bk, wm, wn = gemm_tile(batch, m, n, k, nt)
    align = "".join("a" if d % t == 0 else "u" for d, t in ((m, bm), (n, bn), (k, bk)))
    align = align if align in ALIGNMENTS else "uuu"
    return f"mlx_gemm_{layout}_f16_bm{bm}_bn{bn}_bk{bk}_wm{wm}_wn{wn}_{align}"


def attention_tile(d):
    """(BQ, BK, BD, BV, WM, WN) as MLX's sdpa_full chooses it for f16 and
    head dim d, NAX or not; instantiated in mlx_attention.metal for d = 64."""
    assert d == 64, f"no attention kernel instantiated for head dim {d}"
    if NAX:
        return (64, 32, d, d, 4, 1)
    devc = DEVICE_CLASS
    bk = 32 if (d == 256 and devc == "d") or d < 128 else 16
    return (32, bk, d, d, 4, 2 if d == 256 else 1)


def attention_extern(q, k, v, b, l, h, d, metallib, result="%o", indent="    "):
    """MLIR lines computing `result` = attention(q, k, v) for [B, L, H*D]
    tensors with scale D^-0.5. Returns (lines, result type)."""
    bq, bk, bd, bv, wm, wn = attention_tile(d)
    align = ("a" if l % bq == 0 else "u") + ("a" if l % bk == 0 else "u")
    name = f"mlx_attention{'_nax' if NAX else ''}_f16_bq{bq}_bk{bk}_bd{bd}_bv{bv}_wm{wm}_wn{wn}_{align}"
    t = f"tensor<{b}x{l}x{h * d}xf16>"
    scale_bits = int.from_bytes(struct.pack("<f", d ** -0.5), "little")
    strides = [l * h * d, d, h * d]  # batch, head, row of [B, L, H, D]
    consts = [b, h, d, l, l, scale_bits] + strides * 4
    lines = [f"%c{i} = arith.constant {v} : i32" for i, v in enumerate(consts)]
    lines += [f"%w{i} = arith.constant {v} : index" for i, v in enumerate((b, l, h, d))]
    args = ", ".join(f"%c{i}" for i in range(len(consts)))
    types = ", ".join("i32" for _ in consts)
    lines += [
        f'{result} = hal.dispatch.extern "{name}"[%w0, %w1, %w2, %w3]({args}, {q}, {k}, {v})',
        f"    : ({types}, {t}, {t}, {t}) -> {t}",
        "  count(%device: !hal.device, %wb: index, %wl: index, %wh: index, %wd: index)"
        " -> (index, index, index) {",
        f"    %x = arith.constant {-(-l // bq)} : index",
        f"    %y = arith.constant {h} : index",
        f"    %z = arith.constant {b} : index",
        "    hal.return %x, %y, %z : index, index, index",
        "  }",
        f"  layout(#hal.pipeline.layout<constants = {len(consts)}, bindings = [",
        "    #hal.pipeline.binding<storage_buffer, ReadOnly>,",
        "    #hal.pipeline.binding<storage_buffer, ReadOnly>,",
        "    #hal.pipeline.binding<storage_buffer, ReadOnly>,",
        "    #hal.pipeline.binding<storage_buffer>",
        "  ]>)",
        "  objects({",
        f'    #metal_target ordinal(0) = [#hal.executable.object<{{path = "{metallib}"}}>]',
        "  })",
        f"  <{{workgroup_size = [32 : index, {wm} : index, {wn} : index]}}>",
    ]
    return [indent + line for line in lines], t


def gemm_extern(a, b, batch, m, n, k, metallib, nt=False, a_type=None, b_type=None, indent="    "):
    """MLIR lines computing %d = a * b (f16, rank-3 or rank-2 when batch is
    None) with an extern MLX dispatch. Returns (lines, result type)."""
    prefix = "" if batch is None else f"{batch}x"
    a_type = a_type or f"tensor<{prefix}{m}x{k}xf16>"
    b_type = b_type or (f"tensor<{prefix}{n}x{k}xf16>" if nt else f"tensor<{prefix}{k}x{n}xf16>")
    d_type = f"tensor<{prefix}{m}x{n}xf16>"
    lda, ldb, ldd = k, k if nt else n, n
    if NAX:
        # MLX's GEMMParams, int64 batch strides as (low, high) words, padded
        # to its 8-byte alignment. batch_ndim is unused without batching.
        bm, bn, bk, wm, wn = gemm_nax_tile(m, n, k)
        swizzle = gemm_nax_swizzle(m, n, bm)
        tiles_m, tiles_n = -(-m // bm), -(-n // bn)
        consts = [m, n, k, lda, ldb, ldd, tiles_n, tiles_m, m * k, 0, n * k, 0, m * n, 0,
                  swizzle, k // bk, 1, 0]
    else:
        bm, bn, _, wm, wn = gemm_tile(batch or 1, m, n, k, nt)
        swizzle = SWIZZLE
        tiles_m, tiles_n = -(-m // bm), -(-n // bn)
        consts = [m, n, k, lda, ldb, ldd, m * k, n * k, m * n, swizzle]
    grid = (tiles_n << swizzle, -(-tiles_m // (1 << swizzle)), batch or 1)
    lines = [f"%c{i} = arith.constant {v} : i32" for i, v in enumerate(consts)]
    # The workload is the problem size. The grid does not need it, but IREE
    # keys per-dispatch benchmarks on it, so each shape is benchmarked on its
    # own even when shapes share one kernel (and so one executable).
    workload = [batch or 1, m, n, k]
    lines += [f"%w{i} = arith.constant {v} : index" for i, v in enumerate(workload)]
    args = ", ".join(f"%c{i}" for i in range(len(consts)))
    types = ", ".join("i32" for _ in consts)
    lines += [
        f'%d = hal.dispatch.extern "{kernel_name(batch or 1, m, n, k, nt)}"[%w0, %w1, %w2, %w3]({args}, {a}, {b})',
        f"    : ({types}, {a_type}, {b_type}) -> {d_type}",
        "  count(%device: !hal.device, %wb: index, %wm: index, %wn: index, %wk: index)"
        " -> (index, index, index) {",
        f"    %x = arith.constant {grid[0]} : index",
        f"    %y = arith.constant {grid[1]} : index",
        f"    %z = arith.constant {grid[2]} : index",
        "    hal.return %x, %y, %z : index, index, index",
        "  }",
        f"  layout(#hal.pipeline.layout<constants = {len(consts)}, bindings = [",
        "    #hal.pipeline.binding<storage_buffer, ReadOnly>,",
        "    #hal.pipeline.binding<storage_buffer, ReadOnly>,",
        "    #hal.pipeline.binding<storage_buffer>",
        "  ]>)",
        "  objects({",
        f'    #metal_target ordinal(0) = [#hal.executable.object<{{path = "{metallib}"}}>]',
        "  })",
        f"  <{{workgroup_size = [32 : index, {wn} : index, {wm} : index]}}>",
    ]
    return [indent + line for line in lines], d_type


def rule(name, args, dag, body, acc):
    """A matcher for the DAG over `args` (rooted at its last op) and an action
    that replaces the root with a call to a helper computing `body` from the
    same args. Returns (helper, named sequences, foreach_match rule)."""
    params = ", ".join(f"%{n}: {t}" for n, t in args)
    helper = "\n".join([
        f"  util.func private @{name}({params}) -> {acc} {{",
        *body,
        f"    util.return %r : {acc}",
        "  }",
    ])
    dag = "\n".join(f"        {line}" for line in dag)
    sequences = f"""\
  transform.named_sequence @match_{name}(%root: !transform.any_op {{transform.readonly}})
      -> (!transform.any_value, !transform.any_value) {{
    %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %root {{
      ^bb0({params}):
{dag}
    }} : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    transform.yield %ins, %outs : !transform.any_value, !transform.any_value
  }}
  transform.named_sequence @call_{name}(%ins: !transform.any_value {{transform.readonly}},
                                        %out: !transform.any_value {{transform.readonly}}) {{
    %root = transform.get_defining_op %out : (!transform.any_value) -> !transform.any_op
    %module = transform.util.get_nearest_symbol_table %root : (!transform.any_op) -> !transform.any_op
    %func = transform.util.import_symbol @{name} into %module if undefined
        : (!transform.any_op) -> !transform.any_op
    transform.util.cast_and_call %func(%ins) -> %out after %root {{
    }} : (!transform.any_op, !transform.any_value, !transform.any_value, !transform.any_op) -> !transform.any_op
    transform.yield
  }}"""
    return helper, sequences, f"@match_{name} -> @call_{name}"


def gemm_rule(batch, m, n, k, metallib):
    """fill(0) + batch_matmul(A[b,m,k], B[b,k,n])."""
    a, b = f"tensor<{batch}x{m}x{k}xf16>", f"tensor<{batch}x{k}x{n}xf16>"
    acc = f"tensor<{batch}x{m}x{n}xf32>"
    body, d_type = gemm_extern("%a", "%b", batch, m, n, k, metallib)
    return rule(f"mlx_gemm_{batch}x{m}x{n}x{k}", [("a", a), ("b", b), ("init", acc)], [
        "%zero = arith.constant 0.0 : f32",
        f"%fill = linalg.fill ins(%zero : f32) outs(%init : {acc}) -> {acc}",
        f"%mm = linalg.batch_matmul ins(%a, %b : {a}, {b}) outs(%fill : {acc}) -> {acc}",
    ], body + [f"    %r = arith.extf %d : {d_type} to {acc}"], acc)


def linear_rule(m, n, k, metallib):
    """A torch linear without bias as it reaches preprocessing: the weight
    W[n,k] is transposed, broadcast to batch 1, then fill(0) + batch_matmul.
    The nt kernel reads W directly, so no transposed copy of the weight is
    made (IREE folds that copy at compile time, but not when the consumer is
    an extern dispatch)."""
    a, w = f"tensor<1x{m}x{k}xf16>", f"tensor<{n}x{k}xf16>"
    wt, wb = f"tensor<{k}x{n}xf16>", f"tensor<1x{k}x{n}xf16>"
    acc = f"tensor<1x{m}x{n}xf32>"
    body, d_type = gemm_extern("%a", "%w", 1, m, n, k, metallib, nt=True, b_type=w)
    return rule(f"mlx_linear_{m}x{n}x{k}",
                [("a", a), ("w", w), ("wt_init", wt), ("wb_init", wb), ("init", acc)], [
        f"%wt = linalg.transpose ins(%w : {w}) outs(%wt_init : {wt}) permutation = [1, 0]",
        "%wb = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d1, d2)>, "
        "affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = [\"parallel\", \"parallel\", \"parallel\"]}"
        f" ins(%wt : {wt}) outs(%wb_init : {wb}) {{",
        "^bb0(%in: f16, %out: f16):",
        "  linalg.yield %in : f16",
        f"}} -> {wb}",
        "%zero = arith.constant 0.0 : f32",
        f"%fill = linalg.fill ins(%zero : f32) outs(%init : {acc}) -> {acc}",
        f"%mm = linalg.batch_matmul ins(%a, %wb : {a}, {wb}) outs(%fill : {acc}) -> {acc}",
    ], body + [f"    %r = arith.extf %d : {d_type} to {acc}"], acc)


def attention_rule(b, l, h, d, metallib):
    """q, k, v [b, l, h*d] -> heads [b*h, l, d] -> attention -> [b, l, h*d]."""
    t, heads = f"tensor<{b}x{l}x{h * d}xf16>", f"tensor<{b}x{l}x{h}x{d}xf16>"
    bh, flat = f"tensor<{b}x{h}x{l}x{d}xf16>", f"tensor<{b * h}x{l}x{d}xf16>"
    maps = ", ".join(f"affine_map<(d0, d1, d2, d3, d4) -> ({x})>"
                     for x in ("d0, d1, d3", "d0, d4, d3", "d0, d4, d2", "", "d0, d1, d2"))
    dag = []
    for x in "qkv":
        dag += [
            f"%{x}e = tensor.expand_shape %{x} [[0], [1], [2, 3]] output_shape [{b}, {l}, {h}, {d}]"
            f" : {t} into {heads}",
            f"%{x}t = linalg.transpose ins(%{x}e : {heads}) outs(%heads_init : {bh}) permutation = [0, 2, 1, 3]",
            f"%{x}c = tensor.collapse_shape %{x}t [[0, 1], [2], [3]] : {bh} into {flat}",
        ]
    dag += [
        f"%scale = arith.constant {d ** -0.5:e} : f16",
        f"%a = iree_linalg_ext.attention <indexing_maps = [{maps}]>"
        f" ins(%qc, %kc, %vc, %scale : {flat}, {flat}, {flat}, f16) outs(%attn_init : {flat}) {{",
        "^bb0(%s: f32):",
        "  iree_linalg_ext.yield %s : f32",
        f"}} -> {flat}",
        f"%ae = tensor.expand_shape %a [[0, 1], [2], [3]] output_shape [{b}, {h}, {l}, {d}] : {flat} into {bh}",
        f"%at = linalg.transpose ins(%ae : {bh}) outs(%back_init : {heads}) permutation = [0, 2, 1, 3]",
        f"%out = tensor.collapse_shape %at [[0], [1], [2, 3]] : {heads} into {t}",
    ]
    body, _ = attention_extern("%q", "%k", "%v", b, l, h, d, metallib, result="%r")
    args = [("q", t), ("k", t), ("v", t), ("heads_init", bh), ("attn_init", flat), ("back_init", heads)]
    return rule(f"mlx_attention_{b}x{l}x{h}x{d}", args, dag, body, t)


def spec(rules):
    helpers, sequences, matches = zip(*rules)
    return f"""// Generated by compiler/targets/metal/mlx_kernels.py.
#metal_target = {METAL_TARGET}
module attributes {{transform.with_named_sequence}} {{
{chr(10).join(helpers)}

{chr(10).join(sequences)}

  transform.named_sequence @__transform_main(%module: !transform.any_op) {{
    %funcs = transform.structured.match ops{{["util.func"]}} in %module : (!transform.any_op) -> !transform.any_op
    transform.foreach %funcs : !transform.any_op {{
      ^bb1(%func: !transform.any_op):
        transform.foreach_match in %func
            {(chr(10) + "            , ").join(matches)}
          : (!transform.any_op) -> (!transform.any_op)
    }}
    transform.apply_dce to %module : !transform.any_op
    transform.apply_registered_pass "inline" to %module : (!transform.any_op) -> !transform.any_op
    transform.yield
  }}
}}
"""


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("metallib", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--linear", action="append", default=[],
                   help="MxNxK of a bias-free torch linear (transposed weight) to replace; tried first")
    p.add_argument("--gemm", action="append", default=[], help="BxMxNxK of a batch_matmul to replace")
    p.add_argument("--attention", action="append", default=[],
                   help="BxLxHxD of a torch multi-head attention (scale D^-0.5) to replace")
    p.add_argument("--arch", help="Metal GPU architecture to pick MLX tiles for, e.g. applegpu_g14s "
                   "(default: this machine's GPU)")
    p.add_argument("--nax", choices=["auto", "on", "off"], default="auto",
                   help="NAX GEMMs and attention: as MLX decides from the architecture (auto), or forced")
    a = p.parse_args()
    set_arch(a.arch or detect_arch(), {"auto": None, "on": True, "off": False}[a.nax])
    lib = a.metallib.resolve()
    shape = lambda s: tuple(int(x) for x in s.split("x"))
    rules = ([linear_rule(*shape(s), lib) for s in a.linear] + [gemm_rule(*shape(s), lib) for s in a.gemm]
             + [attention_rule(*shape(s), lib) for s in a.attention])
    if not rules:
        p.error("give at least one --linear, --gemm or --attention")
    a.out.write_text(spec(rules))


if __name__ == "__main__":
    main()
