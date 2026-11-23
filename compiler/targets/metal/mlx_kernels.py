"""Route GEMMs to MLX steel kernels (kernels/mlx_gemm.metal) on IREE Metal.

Writes a preprocessing transform spec (--iree-preprocessing-transform-spec-
filename) that replaces each `fill(0.0 : f32)` + `linalg.batch_matmul` of
f16 operands with the given static shapes by a hal.dispatch.extern call into
the metallib. The kernel writes f16 after f32 accumulation, which is what
IREE's own matmul + truncf computes; the result is extended back to f32 so the
original consumers (truncf + bias/residual epilogue) stay unchanged and get
fused by IREE.

External Metal variants need the patched MetalSPIRV target.

  mlx_kernels.py <mlx_gemm.metallib> <spec.mlir> --gemm BxMxNxK [--gemm ...]
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
# MLX's tile choice for medium Apple GPUs (M1/M2/M3 Pro and Max).
BM, BN, BK, WM, WN, SWIZZLE = 64, 64, 16, 2, 2, 2
ALIGNMENTS = {"aaa", "uaa", "aua", "uua"}  # instantiated, plus "uuu"


def kernel_name(m, n, k, nt=False):
    align = "".join("a" if d % t == 0 else "u" for d, t in ((m, BM), (n, BN), (k, BK)))
    align = align if align in ALIGNMENTS else "uuu"
    return f"mlx_gemm_{'nt' if nt else 'nn'}_f16_bm{BM}_bn{BN}_bk{BK}_wm{WM}_wn{WN}_{align}"


def gemm_extern(a, b, batch, m, n, k, metallib, nt=False, a_type=None, b_type=None, indent="    "):
    """MLIR lines computing %d = a * b (f16, rank-3 or rank-2 when batch is
    None) with an extern MLX dispatch. Returns (lines, result type)."""
    prefix = "" if batch is None else f"{batch}x"
    a_type = a_type or f"tensor<{prefix}{m}x{k}xf16>"
    b_type = b_type or (f"tensor<{prefix}{n}x{k}xf16>" if nt else f"tensor<{prefix}{k}x{n}xf16>")
    d_type = f"tensor<{prefix}{m}x{n}xf16>"
    tiles_m, tiles_n = -(-m // BM), -(-n // BN)
    grid = (tiles_n << SWIZZLE, -(-tiles_m // (1 << SWIZZLE)), batch or 1)
    consts = [m, n, k, k, k if nt else n, n, m * k, n * k, m * n]
    lines = [f"%c{i} = arith.constant {v} : i32" for i, v in enumerate(consts)]
    args = ", ".join(f"%c{i}" for i in range(len(consts)))
    types = ", ".join("i32" for _ in consts)
    lines += [
        f'%d = hal.dispatch.extern "{kernel_name(m, n, k, nt)}"({args}, {a}, {b})',
        f"    : ({types}, {a_type}, {b_type}) -> {d_type}",
        "  count(%device: !hal.device) -> (index, index, index) {",
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
        f"  <{{workgroup_size = [32 : index, {WN} : index, {WM} : index]}}>",
    ]
    return [indent + line for line in lines], d_type


def spec(gemms, metallib):
    helpers, matchers, rules = [], [], []
    for batch, m, n, k in gemms:
        tag = f"{batch}x{m}x{n}x{k}"
        a, b = f"tensor<{batch}x{m}x{k}xf16>", f"tensor<{batch}x{k}x{n}xf16>"
        acc = f"tensor<{batch}x{m}x{n}xf32>"
        body, d_type = gemm_extern("%a", "%b", batch, m, n, k, metallib)
        helpers.append("\n".join([
            f"  util.func private @mlx_gemm_{tag}(%a: {a}, %b: {b}, %init: {acc}) -> {acc} {{",
            *body,
            f"    %r = arith.extf %d : {d_type} to {acc}",
            f"    util.return %r : {acc}",
            "  }",
        ]))
        matchers.append(f"""\
  transform.named_sequence @match_{tag}(%root: !transform.any_op {{transform.readonly}})
      -> (!transform.any_value, !transform.any_value) {{
    %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %root {{
      ^bb0(%a: {a}, %b: {b}, %init: {acc}):
        %zero = arith.constant 0.0 : f32
        %fill = linalg.fill ins(%zero : f32) outs(%init : {acc}) -> {acc}
        %mm = linalg.batch_matmul ins(%a, %b : {a}, {b}) outs(%fill : {acc}) -> {acc}
    }} : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    transform.yield %ins, %outs : !transform.any_value, !transform.any_value
  }}
  transform.named_sequence @call_{tag}(%ins: !transform.any_value {{transform.readonly}},
                                       %out: !transform.any_value {{transform.readonly}}) {{
    %root = transform.get_defining_op %out : (!transform.any_value) -> !transform.any_op
    %module = transform.util.get_nearest_symbol_table %root : (!transform.any_op) -> !transform.any_op
    %func = transform.util.import_symbol @mlx_gemm_{tag} into %module if undefined
        : (!transform.any_op) -> !transform.any_op
    transform.util.cast_and_call %func(%ins) -> %out after %root {{
    }} : (!transform.any_op, !transform.any_value, !transform.any_value, !transform.any_op) -> !transform.any_op
    transform.yield
  }}""")
        rules.append(f"@match_{tag} -> @call_{tag}")
    return f"""// Generated by compiler/targets/metal/mlx_kernels.py.
#metal_target = {METAL_TARGET}
module attributes {{transform.with_named_sequence}} {{
{chr(10).join(helpers)}

{chr(10).join(matchers)}

  transform.named_sequence @__transform_main(%module: !transform.any_op) {{
    %funcs = transform.structured.match ops{{["util.func"]}} in %module : (!transform.any_op) -> !transform.any_op
    transform.foreach %funcs : !transform.any_op {{
      ^bb1(%func: !transform.any_op):
        transform.foreach_match in %func
            {(chr(10) + "            , ").join(rules)}
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
    p.add_argument("--gemm", action="append", required=True, help="BxMxNxK of a batch_matmul to replace")
    a = p.parse_args()
    gemms = [tuple(int(x) for x in g.split("x")) for g in a.gemm]
    a.out.write_text(spec(gemms, a.metallib.resolve()))


if __name__ == "__main__":
    main()
