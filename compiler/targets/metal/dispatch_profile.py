"""Per-dispatch GPU time breakdown of an IREE Metal module.

IREE's Metal HAL has no device profiling (--device_profiling_mode returns
UNIMPLEMENTED), so this times each executable on its own and weights it by how
often the model dispatches it:

  1. lower the input to stream IR and count stream.cmd.dispatch sites per
     (executable, export, workload);
  2. dump standalone benchmark modules: one per executable, with one function
     per distinct workload;
  3. compile and run each with iree-benchmark-module, one at a time.

External dispatches (hal.dispatch.extern) of different shapes can share one
executable; they are told apart by their workload, which mlx_kernels.py sets
to the problem size.

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
# @exe::@export for generated executables, @exe::@variant::@export for
# external ones; then the workload when there is one. The printer names index
# constants %c<value> (%c<value>_<n> when repeated).
SITE = re.compile(r"stream\.cmd\.dispatch @([^:\s]+)::(?:@[^:\s]+::)?@([^\s(\[]+)(?:\[([^\]]*)\])?")


def run(*cmd, **kw):
    return subprocess.run([str(c) for c in cmd], check=True, text=True, capture_output=True, **kw).stdout


def workload(text):
    if not text:
        return ""
    values = [re.fullmatch(r"%c(\d+)(?:_\d+)?", v.strip()) for v in text.split(",")]
    return "x".join(v.group(1) for v in values) if all(values) else ""


def dispatch_counts(src, work, flags):
    stream = work / "stream.mlir"
    run(COMPILER, src, *flags, "--compile-to=stream", "--mlir-elide-resource-strings-if-larger=16", "-o", stream)
    # Initializers run once at load and are not counted.
    return collections.Counter((exe, export, workload(wl)) for exe, export, wl in SITE.findall(stream.read_text())
                               if not exe.startswith("_initializer"))


def bench_file(mlir, work, batch):
    """Returns {benchmark function name: us per dispatch}."""
    fixed = work / mlir.name.replace("$", "_")
    fixed.write_text(TRANSLATION_INFO.sub("", mlir.read_text()))
    vmfb = fixed.with_suffix(".vmfb")
    run(COMPILER, fixed, "-o", vmfb)
    # iree-benchmark-module runs every benchmark function and reports time
    # per batch item.
    out = run(BIN / "iree-benchmark-module", "--device=metal", f"--module={vmfb}", f"--batch_size={batch}",
              "--benchmark_repetitions=5", "--benchmark_min_time=0.3s", "--benchmark_report_aggregates_only=true")
    times = {}
    for name, value, unit in re.findall(r"BM_(\S+?)/\S*real_time_median\s+([\d.]+) (\w+)", out):
        times[name] = float(value) * {"ns": 1e-3, "us": 1, "ms": 1e3}[unit]
    return times


def flops(kind, wl):
    if kind.startswith("mlx_gemm") and wl.count("x") == 3:
        b, m, n, k = (int(x) for x in wl.split("x"))
        return 2 * b * m * n * k
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
        for name, us in bench_file(mlir, a.work_dir, a.batch).items():
            times[name] = us
            print(f"  {name} {us:.1f} us", file=sys.stderr)

    rows = []
    for (exe, export, wl), n in counts.items():
        base = f"{exe}_metal_msl_fb_{export}"
        us = times.get(f"{base}_{wl}", times.get(base))
        if us is None:
            sys.exit(f"no benchmark for {base} (workload {wl or 'none'})")
        m = re.match(r"main\$async_(dispatch_\d+)_(\w+)", export)
        d, kind = m.groups() if m else (exe, export)
        rows.append((n * us, n, us, d, kind, wl))
    total = sum(r[0] for r in rows)
    print(f"{'dispatch':17s} {'kind':60s} {'calls':>5s} {'each':>9s} {'total':>9s} {'share':>6s}  rate")
    for t, n, us, d, kind, wl in sorted(rows, reverse=True):
        f = flops(kind, wl)
        rate = f"{f / us / 1e6:.2f} TFLOPS" if f else ""
        label = f"{kind} {wl}" if kind.startswith("mlx_") else kind
        print(f"{d:17s} {label:60s} {n:5d} {us / 1e3:6.2f} ms {t / 1e3:6.1f} ms {100 * t / total:5.1f}%  {rate}")
    print(f"total {total / 1e3:.1f} ms over {sum(counts.values())} dispatches")


if __name__ == "__main__":
    main()
