"""Check an exported BioCLIP 2 encoder against the fp32 reference.

Runs the artifact on the reference pixels (pixels.npy from `export.py
reference`), then compares, per photo, the embedding with ref.fp32.npy
(cosine) and the TreeOfLife top-5 with ref.top5.json.

  compare.py <ref_dir> <dataset_dir> <artifact> [--device metal] [--function b1]

<artifact> is an IREE .vmfb (batch read from its vision.b<N>. name), a Core ML
.mlpackage (run with compute units ALL) or an ONNX model. Exits non-zero
when any embedding has cosine < 0.999 or any top-1 differs.
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

import bioclip


def run_iree(path, pixels, device):
    import iree.runtime as rt

    config = rt.Config(device)
    vmfb = rt.VmModule.mmap(config.vm_instance, str(path))
    ctx = rt.SystemContext(config=config, vm_modules=[vmfb])
    fn = ctx.modules.module["main"]
    # The vmfb carries no ABI reflection; export.py names it vision.b<N>.*.
    batch = int(re.search(r"\.b(\d+)\.", path.name).group(1))
    x = pixels.astype(np.float16)
    outs, times = [], []
    for i in range(0, len(x), batch):
        chunk = x[i:i + batch]
        if len(chunk) < batch:
            chunk = np.concatenate([chunk, np.zeros((batch - len(chunk),) + chunk.shape[1:], chunk.dtype)])
        t = time.perf_counter()
        y = fn(chunk).to_host()
        times.append(time.perf_counter() - t)
        outs.append(y)
    return np.concatenate(outs)[:len(x)].astype(np.float32), times


def run_coreml(path, pixels, function):
    import coremltools as ct

    model = ct.models.MLModel(str(path), compute_units=ct.ComputeUnit.ALL, function_name=function)
    desc = model.get_spec().description
    if function:  # a multifunction package describes inputs per function
        desc = next(f for f in desc.functions if f.name == function)
    batch = desc.input[0].type.multiArrayType.shape[0]
    outs, times = [], []
    for i in range(0, len(pixels), batch):
        chunk = pixels[i:i + batch]
        if len(chunk) < batch:
            chunk = np.concatenate([chunk, np.zeros((batch - len(chunk),) + chunk.shape[1:], chunk.dtype)])
        t = time.perf_counter()
        y = model.predict({"pixel_values": chunk})["image_features"]
        times.append(time.perf_counter() - t)
        outs.append(y)
    return np.concatenate(outs)[:len(pixels)].astype(np.float32), times


def run_onnx(path, pixels):
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path))
    outs, times = [], []
    for i in range(len(pixels)):
        t = time.perf_counter()
        outs.append(sess.run(None, {"pixel_values": pixels[i:i + 1]})[0])
        times.append(time.perf_counter() - t)
    return np.concatenate(outs).astype(np.float32), times


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ref_dir")
    p.add_argument("dataset_dir")
    p.add_argument("artifact")
    p.add_argument("--device", default="metal")
    p.add_argument("--function", default=None, help="Core ML multifunction name, e.g. b1")
    a = p.parse_args()

    ref_dir = Path(a.ref_dir)
    pixels = np.load(ref_dir / "pixels.npy")
    ref = np.load(ref_dir / "ref.fp32.npy")
    ref_top = json.loads((ref_dir / "ref.top5.json").read_text())

    art = Path(a.artifact)
    if art.suffix == ".vmfb":
        emb, times = run_iree(art, pixels, a.device)
    elif art.suffix == ".mlpackage":
        emb, times = run_coreml(art, pixels, a.function)
    else:
        emb, times = run_onnx(art, pixels)

    cos = (emb * ref).sum(1) / np.linalg.norm(emb, axis=1) / np.linalg.norm(ref, axis=1)
    tol = bioclip.TreeOfLife(a.dataset_dir)
    top = tol.top_k(emb, 5)
    ref_sims = tol.similarities(ref)
    same1 = same5 = 0
    swap_gap = 0.0  # worst reference-cosine gap between an entry dropped from and one added to the top-5
    for j, (r, hits) in enumerate(zip(ref_top, top)):
        want = [h["index"] for h in r["top5"]]
        got = [i for i, _, _ in hits]
        same1 += want[0] == got[0]
        same5 += set(want) == set(got)
        dropped, added = set(want) - set(got), set(got) - set(want)
        if dropped:
            swap_gap = max(swap_gap, max(ref_sims[j, d] for d in dropped) - min(ref_sims[j, x] for x in added))
    n = len(ref_top)
    warm = times[1:] or times
    print(f"{art.name}: cosine min {cos.min():.5f} mean {cos.mean():.5f} | "
          f"top-1 same {same1}/{n} | top-5 set same {same5}/{n} (worst swap gap {swap_gap:.4f}) | "
          f"call median {np.median(warm) * 1e3:.1f} ms (first {times[0] * 1e3:.0f} ms, not a benchmark)")
    sys.exit(0 if cos.min() >= 0.999 and same1 == n else 1)


if __name__ == "__main__":
    main()
