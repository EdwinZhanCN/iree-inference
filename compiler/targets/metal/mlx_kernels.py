"""Route GEMMs to MLX steel kernels (kernels/mlx_gemm.metal) on IREE Metal.

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

  mlx_kernels.py <mlx_gemm.metallib> <spec.mlir> [--linear MxNxK]... [--gemm BxMxNxK]...
"""
import argparse
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
# (BM, BN, BK, WM, WN) per layout; must be instantiated in mlx_gemm.metal.
# nn is MLX's choice for medium Apple GPUs (M1/M2/M3 Pro and Max). For nt,
# 32x32x32 was the fastest on an M2 Pro for the BioCLIP 2 linears (M = 257)
# among MLX's presets (64x64x16, 64x32x32, 32x64x16) and 32x32x16, 32x64x32,
# 64x64x32: the three linear shapes take 42.0 ms per encoder call against
# 47.8 ms with 64x32x32 and BM = 32 wastes less of the last tile row.
TILES = {"nn": (64, 64, 16, 2, 2), "nt": (32, 32, 32, 2, 2)}
SWIZZLE = 2
ALIGNMENTS = {"aaa", "uaa", "aua", "uua"}  # instantiated, plus "uuu"


def kernel_name(m, n, k, nt=False):
    layout = "nt" if nt else "nn"
    bm, bn, bk, wm, wn = TILES[layout]
    align = "".join("a" if d % t == 0 else "u" for d, t in ((m, bm), (n, bn), (k, bk)))
    align = align if align in ALIGNMENTS else "uuu"
    return f"mlx_gemm_{layout}_f16_bm{bm}_bn{bn}_bk{bk}_wm{wm}_wn{wn}_{align}"


def gemm_extern(a, b, batch, m, n, k, metallib, nt=False, a_type=None, b_type=None, indent="    "):
    """MLIR lines computing %d = a * b (f16, rank-3 or rank-2 when batch is
    None) with an extern MLX dispatch. Returns (lines, result type)."""
    prefix = "" if batch is None else f"{batch}x"
    a_type = a_type or f"tensor<{prefix}{m}x{k}xf16>"
    b_type = b_type or (f"tensor<{prefix}{n}x{k}xf16>" if nt else f"tensor<{prefix}{k}x{n}xf16>")
    d_type = f"tensor<{prefix}{m}x{n}xf16>"
    bm, bn, _, wm, wn = TILES["nt" if nt else "nn"]
    tiles_m, tiles_n = -(-m // bm), -(-n // bn)
    grid = (tiles_n << SWIZZLE, -(-tiles_m // (1 << SWIZZLE)), batch or 1)
    consts = [m, n, k, k, k if nt else n, n, m * k, n * k, m * n]
    lines = [f"%c{i} = arith.constant {v} : i32" for i, v in enumerate(consts)]
    # The workload is the problem size. The grid does not need it, but IREE
    # keys per-dispatch benchmarks on it, so each shape is benchmarked on its
    # own even when shapes share one kernel (and so one executable).
    workload = [batch or 1, m, n, k]
    lines += [f"%w{i} = arith.constant {v} : index" for i, v in enumerate(workload)]
    args = ", ".join(f"%c{i}" for i in range(len(consts)))
    types = ", ".join("i32" for _ in consts)
    lines += [
        f'%d = hal.dispatch.extern "{kernel_name(m, n, k, nt)}"[%w0, %w1, %w2, %w3]({args}, {a}, {b})',
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
    a = p.parse_args()
    lib = a.metallib.resolve()
    shape = lambda s: tuple(int(x) for x in s.split("x"))
    rules = [linear_rule(*shape(s), lib) for s in a.linear] + [gemm_rule(*shape(s), lib) for s in a.gemm]
    if not rules:
        p.error("give at least one --linear or --gemm")
    a.out.write_text(spec(rules))


if __name__ == "__main__":
    main()
