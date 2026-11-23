"""Per-dispatch GPU time breakdown of an IREE Metal module.

IREE's Metal HAL has no device profiling (--device_profiling_mode returns
UNIMPLEMENTED), so this times each executable on its own and weights it by how
often the model dispatches it:

  1. lower the input to stream IR and count stream.cmd.dispatch sites;
  2. dump one standalone benchmark module per executable;
  3. compile and run each with iree-benchmark-module, one at a time.

The dumped modules keep translation_info but lose lowering_config, which
makes matmuls fail to compile; translation_info is stripped so configuration
runs again with the same heuristics as the full compile. The weighted sum
should land close to the end-to-end call time.

  dispatch_profile.py <input.mlirbc> <work_dir> [--compiler PATH] [iree-compile flags...]
"""
import argparse
import collections
import re
import shutil
import subprocess
import sys
from pathlib import Path

BIN = Path(sys.executable).parent
COMPILER = BIN / "iree-compile"
TRANSLATION_INFO = re.compile(r" attributes \{translation_info = #iree_codegen\.translation_info<[^}]*>\}")
# FLOPs for matmul-like executables, from the shape in the executable name.
MATMUL = re.compile(r"(batch_)?matmul_(?:(\d+)x)?(\d+)x(\d+)x(\d+)_")


def run(*cmd, **kw):
    return subprocess.run([str(c) for c in cmd], check=True, text=True, capture_output=True, **kw).stdout


def dispatch_counts(src, work, flags):
    stream = work / "stream.mlir"
    run(COMPILER, src, *flags, "--compile-to=stream", "-o", stream)
    # @exe::@export for generated executables, @exe::@variant::@export for
    # external ones. Initializers run once at load and are not counted.
    sites = re.findall(r"stream\.cmd\.dispatch @([^:\s]+)::(?:@[^:\s]+::)?@([^\s(]+)", stream.read_text())
    return collections.Counter((exe, export) for exe, export in sites if not exe.startswith("_initializer"))


def bench_one(mlir, work, batch):
    name = re.match(r"module_(.+)_metal_msl_fb_benchmark\.mlir", mlir.name).group(1)
    fixed = work / f"{name}.mlir".replace("$", "_")
    fixed.write_text(TRANSLATION_INFO.sub("", mlir.read_text()))
    vmfb = fixed.with_suffix(".vmfb")
    run(COMPILER, fixed, "-o", vmfb)
    # iree-benchmark-module reports time per batch item.
    out = run(BIN / "iree-benchmark-module", "--device=metal", f"--module={vmfb}", f"--batch_size={batch}",
              "--benchmark_repetitions=5", "--benchmark_min_time=0.3s", "--benchmark_report_aggregates_only=true")
    value, unit = re.search(r"real_time_median\s+([\d.]+) (\w+)", out).groups()
    return name, float(value) * {"ns": 1e-3, "us": 1, "ms": 1e3}[unit]


def flops(kind):
    m = MATMUL.search(kind + "_")
    if not m:
        return None
    b, mm, n, k = (int(x) if x else 1 for x in m.groups()[1:])
    return 2 * b * mm * n * k


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", type=Path)
    p.add_argument("work_dir", type=Path)
    p.add_argument("--batch", type=int, default=100, help="dispatches per command buffer")
    p.add_argument("--compiler", type=Path, default=COMPILER, help="iree-compile to use")
    a, flags = p.parse_known_args()
    globals()["COMPILER"] = a.compiler
    flags = flags or ["--iree-hal-target-device=metal"]

    shutil.rmtree(a.work_dir, ignore_errors=True)
    dumps = a.work_dir / "dumps"
    dumps.mkdir(parents=True)
    counts = dispatch_counts(a.input, a.work_dir, flags)
    run(COMPILER, a.input, *flags, f"--iree-hal-dump-executable-benchmarks-to={dumps}",
        "-o", a.work_dir / "full.vmfb")

    times = {}
    for mlir in sorted(dumps.glob("*.mlir")):
        if mlir.name.startswith("module__initializer"):
            continue
        name, us = bench_one(mlir, a.work_dir, a.batch)
        times[name] = us
        print(f"  {name} {us:.1f} us", file=sys.stderr)

    rows = []
    for (exe, export), n in counts.items():
        m = re.match(r"main\$async_(dispatch_\d+)_(\w+)", export)
        d, kind = m.groups() if m else (exe, export)
        rows.append((n * times[exe], n, times[exe], d, kind))
    total = sum(r[0] for r in rows)
    print(f"{'dispatch':17s} {'kind':45s} {'calls':>5s} {'each':>9s} {'total':>9s} {'share':>6s}  rate")
    for t, n, us, d, kind in sorted(rows, reverse=True):
        f = flops(kind)
        rate = f"{f / us / 1e6:.2f} TFLOPS" if f else ""
        print(f"{d:17s} {kind:45s} {n:5d} {us / 1e3:6.2f} ms {t / 1e3:6.1f} ms {100 * t / total:5.1f}%  {rate}")
    print(f"total {total / 1e3:.1f} ms over {sum(counts.values())} dispatches")


if __name__ == "__main__":
    main()
