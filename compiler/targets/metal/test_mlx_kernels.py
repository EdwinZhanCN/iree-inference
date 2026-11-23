"""Check MLX steel kernels called through hal.dispatch.extern on Metal.

For each GEMM shape, builds a module with one extern dispatch, compiles it
with a patched iree-compile (external Metal variants), checks it against NumPy
and IREE's own matmul, and reports the median call time from
iree-benchmark-module. Attention shapes are checked against NumPy only.

  test_mlx_kernels.py <iree-compile> <mlx_kernels.metallib> <work_dir>
      [--shape MxNxK[:nt]]... [--attention BxLxHxD]...
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

import mlx_kernels
from mlx_kernels import METAL_TARGET, attention_extern, gemm_extern, kernel_name

BIN = Path(sys.executable).parent


def module(m, n, k, nt, metallib):
    body, d_type = gemm_extern("%a", "%b", None, m, n, k, metallib, nt=nt)
    b_shape = f"{n}x{k}" if nt else f"{k}x{n}"
    lines = "\n".join(body)
    return f"""
#metal_target = {METAL_TARGET}
#device = #hal.device.target<"metal", [#metal_target]> : !hal.device
module attributes {{hal.device.targets = [#device]}} {{
  func.func @main(%a: tensor<{m}x{k}xf16>, %b: tensor<{b_shape}xf16>) -> {d_type} {{
{lines}
    return %d : {d_type}
  }}
}}
"""


def attention_module(b, l, h, d, metallib):
    body, t = attention_extern("%q", "%k", "%v", b, l, h, d, metallib)
    lines = "\n".join(body)
    return f"""
#metal_target = {METAL_TARGET}
#device = #hal.device.target<"metal", [#metal_target]> : !hal.device
module attributes {{hal.device.targets = [#device]}} {{
  func.func @main(%q: {t}, %k: {t}, %v: {t}) -> {t} {{
{lines}
    return %o : {t}
  }}
}}
"""


def check_attention(rt, iree_compile, metallib, work_dir, spec):
    """MLX attention on [B, L, H*D] q, k, v against NumPy; True when it matches."""
    b, l, h, d = (int(x) for x in spec.split("x"))
    stem = work_dir / f"attention_{spec}"
    stem.with_suffix(".mlx.mlir").write_text(attention_module(b, l, h, d, metallib))
    subprocess.run([iree_compile, stem.with_suffix(".mlx.mlir"), "-o", stem.with_suffix(".mlx.vmfb")], check=True)
    rng = np.random.default_rng(0)
    q, k, v = (rng.standard_normal((b, l, h * d)).astype(np.float16) for _ in range(3))
    heads = lambda x: x.astype(np.float32).reshape(b, l, h, d).transpose(0, 2, 1, 3)
    s = heads(q) @ heads(k).transpose(0, 1, 3, 2) / np.sqrt(d)
    p = np.exp(s - s.max(-1, keepdims=True))
    want = ((p / p.sum(-1, keepdims=True)) @ heads(v)).transpose(0, 2, 1, 3).reshape(b, l, h * d)
    got = run_module(rt, stem.with_suffix(".mlx.vmfb"), q, k, v)
    err = float(np.abs(got - want).max() / np.abs(want).max())
    good = err < 5e-3
    print(f"attention {spec:12s} max rel err: mlx {err:.1e} ({'ok' if good else 'FAIL'})")
    return good


def baseline_module(m, n, k, nt):
    """The same GEMM as linalg.matmul (f16 inputs, f32 accumulation), for
    IREE's own metal-spirv codegen."""
    b_shape = f"{n}x{k}" if nt else f"{k}x{n}"
    b_map = "affine_map<(m, n, k) -> (n, k)>" if nt else "affine_map<(m, n, k) -> (k, n)>"
    return f"""
func.func @main(%a: tensor<{m}x{k}xf16>, %b: tensor<{b_shape}xf16>) -> tensor<{m}x{n}xf16> {{
  %zero = arith.constant 0.0 : f32
  %e = tensor.empty() : tensor<{m}x{n}xf32>
  %acc = linalg.fill ins(%zero : f32) outs(%e : tensor<{m}x{n}xf32>) -> tensor<{m}x{n}xf32>
  %r = linalg.matmul indexing_maps = [affine_map<(m, n, k) -> (m, k)>, {b_map}, affine_map<(m, n, k) -> (m, n)>]
      ins(%a, %b : tensor<{m}x{k}xf16>, tensor<{b_shape}xf16>) outs(%acc : tensor<{m}x{n}xf32>) -> tensor<{m}x{n}xf32>
  %t = arith.truncf %r : tensor<{m}x{n}xf32> to tensor<{m}x{n}xf16>
  return %t : tensor<{m}x{n}xf16>
}}
"""


def bench_us(vmfb, work_dir, stem):
    out = subprocess.run([BIN / "iree-benchmark-module", "--device=metal", f"--module={vmfb}",
                          "--function=main", f"--input=@{stem}.x.npy", f"--input=@{stem}.w.npy",
                          "--benchmark_repetitions=5", "--benchmark_report_aggregates_only=true"],
                         cwd=work_dir, check=True, text=True, capture_output=True).stdout
    value, unit = re.search(r"real_time_median\s+([\d.]+) (\w+)", out).groups()
    return float(value) * {"ns": 1e-3, "us": 1, "ms": 1e3}[unit]


def run_module(rt, vmfb, *args):
    config = rt.Config("metal")
    ctx = rt.SystemContext(config=config)
    ctx.add_vm_module(rt.VmModule.mmap(config.vm_instance, str(vmfb)))
    return np.asarray(ctx.modules.module.main(*args)).astype(np.float32)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("iree_compile", type=Path)
    p.add_argument("metallib", type=Path)
    p.add_argument("work_dir", type=Path)
    p.add_argument("--shape", action="append",
                   help="MxNxK, with :nt for a transposed B (default: the BioCLIP 2 shapes and one unaligned shape)")
    p.add_argument("--attention", action="append",
                   help="BxLxHxD (default: BioCLIP 2's and one with L a multiple of the tile)")
    p.add_argument("--arch", help="Metal GPU architecture to pick MLX tiles for (default: this machine's GPU)")
    a = p.parse_args()
    mlx_kernels.set_arch(a.arch or mlx_kernels.detect_arch())
    # The model's weights reach the dispatch as KxN, so nn.
    a.shape = a.shape or ["257x1024x1024", "257x4096x1024", "257x1024x4096", "100x96x40", "257x1024x1024:nt"]
    a.work_dir = a.work_dir.resolve()
    a.work_dir.mkdir(parents=True, exist_ok=True)
    import iree.runtime as rt

    a.attention = a.attention or ["1x257x16x64", "2x64x4x64"]
    ok = True
    for spec in a.attention:
        ok &= check_attention(rt, a.iree_compile, a.metallib.resolve(), a.work_dir, spec)
    for spec in a.shape:
        dims, _, mode = spec.partition(":")
        m, n, k = (int(x) for x in dims.split("x"))
        nt = mode == "nt"
        stem = a.work_dir / f"gemm_{dims}_{'nt' if nt else 'nn'}"
        mlx_vmfb, base_vmfb = stem.with_suffix(".mlx.vmfb"), stem.with_suffix(".iree.vmfb")
        stem.with_suffix(".mlx.mlir").write_text(module(m, n, k, nt, a.metallib.resolve()))
        stem.with_suffix(".iree.mlir").write_text(baseline_module(m, n, k, nt))
        subprocess.run([a.iree_compile, stem.with_suffix(".mlx.mlir"), "-o", mlx_vmfb], check=True)
        subprocess.run([a.iree_compile, stem.with_suffix(".iree.mlir"), "--iree-hal-target-device=metal",
                        "-o", base_vmfb], check=True)

        rng = np.random.default_rng(0)
        x = rng.standard_normal((m, k)).astype(np.float16)
        w = (rng.standard_normal((n, k) if nt else (k, n)) / np.sqrt(k)).astype(np.float16)
        want = x.astype(np.float32) @ (w.T if nt else w).astype(np.float32)
        err = {}
        for name, vmfb in (("mlx", mlx_vmfb), ("iree", base_vmfb)):
            got = run_module(rt, vmfb, x, w)
            err[name] = float(np.abs(got - want).max() / np.abs(want).max())

        np.save(stem.with_name(stem.name + ".x.npy"), x)
        np.save(stem.with_name(stem.name + ".w.npy"), w)
        us = {name: bench_us(vmfb, a.work_dir, stem.name) for name, vmfb in (("mlx", mlx_vmfb), ("iree", base_vmfb))}
        good = err["mlx"] < 2e-3
        ok &= good
        print(f"{spec:18s} {kernel_name(1, m, n, k, nt)}\n"
              f"    max rel err: mlx {err['mlx']:.1e} ({'ok' if good else 'FAIL'}), iree {err['iree']:.1e}\n"
              f"    call median: mlx {us['mlx'] / 1e3:.3f} ms, iree {us['iree'] / 1e3:.3f} ms "
              f"({us['iree'] / us['mlx']:.1f}x)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
