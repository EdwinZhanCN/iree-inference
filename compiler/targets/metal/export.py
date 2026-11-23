"""Export the BioCLIP 2 image encoder (ViT-L/14) from the upstream open_clip
checkpoint, for IREE and for Core ML, plus the fp32 reference.

All artifacts come from the same upstream weights (hf-hub:imageomics/bioclip-2)
so every configuration in the Metal benchmark runs the same model.

  reference <upstream> <samples_dir> <dataset_dir> <out>
      fp32 PyTorch on the sample photos: pixels.npy (the preprocessed inputs
      every other path is fed), ref.fp32.npy (embeddings, not normalized) and
      ref.top5.json (top-5 TreeOfLife entries per photo).
  iree <upstream> <out> [--batch N] [--res R] [--attention sdpa|decomposed]
      f16 encoder (LayerNorm in f32, see LayerNormF32) with static shapes as MLIR bytecode,
      vision.b<N>.r<R>.f16[.decomposed].mlirbc. With sdpa (the rewriter's
      input) attention imports as iree_linalg_ext.attention, which IREE 3.13's
      metal-spirv codegen cannot compile; the decomposed variant (matmul,
      softmax, matmul) is the default-path baseline.
  coreml <upstream> <out> [--batches 1,8,32] [--res R]
      Core ML tier 2: iOS 18 deployment target (fused SDPA), fp16 compute,
      one mlpackage with a function per batch size sharing the weights,
      vision.r<R>.ios18.mlpackage (functions b1, b8, ...).

Resolutions other than 224 interpolate the position embedding (open_clip's
force_image_size); they are for latency only, the accuracy checks use 224.
"""

import argparse
import gc
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch

import bioclip


class Encoder(torch.nn.Module):
    """`[N, 3, R, R]` pixels -> `[N, 768]` image features, not normalized."""

    def __init__(self, visual):
        super().__init__()
        self.visual = visual

    def forward(self, pixels):
        return self.visual(pixels)


class LayerNormF32(torch.nn.LayerNorm):
    """LayerNorm computed in f32 inside an f16 model.

    IREE lowers an f16 aten.layer_norm entirely in f16: the mean, the sum of
    squares over 1024 channels and the rsqrt. BioCLIP 2's residual stream
    reaches |x| ~ 190 from block 10 on, so the sum of squares is in the tens
    of thousands where f16 steps are 16-32; on some photos the embedding then
    drops to cosine 0.996 against fp32 (PyTorch's f16 LayerNorm accumulates
    in f32 and stays at 1.0). Everything else stays f16.
    """

    def forward(self, x):
        y = torch.nn.functional.layer_norm(
            x.float(), self.normalized_shape, self.weight.float(), self.bias.float(), self.eps)
        return y.to(x.dtype)


def f32_layer_norms(module):
    for name, child in module.named_children():
        if type(child) is not LayerNormF32 and isinstance(child, torch.nn.LayerNorm):
            ln = LayerNormF32(child.normalized_shape, eps=child.eps)
            ln.load_state_dict(child.state_dict())
            setattr(module, name, ln)
        else:
            f32_layer_norms(child)
    return module


def load_encoder(upstream, res, sdpa=True):
    import open_clip

    kwargs = {} if res == 224 else {"force_image_size": res}
    model = open_clip.create_model(f"local-dir:{upstream}", **kwargs)
    for block in model.visual.transformer.resblocks:
        attn = block.attn
        attn.use_sdpa = sdpa
        # open_clip passes scale=head_dim**-0.5 to SDPA, which is SDPA's
        # default. With an explicit scale coremltools decomposes SDPA into
        # matmul/softmax instead of emitting iOS 18's fused op.
        assert abs(attn.scale - attn.head_dim ** -0.5) < 1e-12
        if sdpa:
            attn.scale = None
    return Encoder(model.visual).eval()


def cmd_reference(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    samples = json.loads(Path(__file__).with_name("samples.json").read_text())
    model, preprocess = bioclip.load_model(a.upstream)
    pixels = bioclip.load_images([Path(a.samples_dir) / f"{s['observation']}.jpg" for s in samples], preprocess)
    with torch.no_grad():
        ref = model.encode_image(torch.from_numpy(pixels)).numpy()
    np.save(out / "pixels.npy", pixels)
    np.save(out / "ref.fp32.npy", ref)

    tol = bioclip.TreeOfLife(a.dataset_dir)
    top = tol.top_k(ref, 5)
    records = []
    for s, hits in zip(samples, top):
        records.append({
            "observation": s["observation"],
            "taxon": s["taxon"],
            "top5": [{"index": i, "label": tol.labels[i], "cosine": c, "score": p} for i, c, p in hits],
        })
    (out / "ref.top5.json").write_text(json.dumps(records, indent=1, ensure_ascii=False))
    hit1 = sum(" ".join(r["top5"][0]["label"].split("/")[5:7]) == r["taxon"] for r in records)
    print(f"reference: {len(records)} photos, species top-1 {hit1}/{len(records)}")


def cmd_iree(a):
    import iree.turbine.aot as aot

    enc = f32_layer_norms(load_encoder(a.upstream, a.res, sdpa=a.attention == "sdpa")).half()
    example = torch.zeros(a.batch, 3, a.res, a.res, dtype=torch.float16)
    exported = aot.export(enc, args=(example,))
    suffix = "" if a.attention == "sdpa" else ".decomposed"
    path = Path(a.out) / f"vision.b{a.batch}.r{a.res}.f16{suffix}.mlirbc"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        exported.mlir_module.operation.write_bytecode(f)
    print(path)


def cmd_coreml(a):
    import coremltools as ct

    enc = load_encoder(a.upstream, a.res)
    out = Path(a.out) / f"vision.r{a.res}.ios18.mlpackage"
    batches = [int(b) for b in a.batches.split(",")]
    with tempfile.TemporaryDirectory() as tmp:
        desc = None
        for b in batches:
            # Tracing runs one forward pass; no_grad keeps autograd buffers out.
            example = torch.zeros(b, 3, a.res, a.res)
            with torch.no_grad():
                traced = torch.jit.trace(enc, example)
            # skip_model_load: do not compile and load the model into Core ML
            # after converting (multi-GB for ViT-L, more at batch 32).
            ml = ct.convert(
                traced,
                inputs=[ct.TensorType(name="pixel_values", shape=example.shape, dtype=np.float32)],
                outputs=[ct.TensorType(name="image_features", dtype=np.float16)],
                minimum_deployment_target=ct.target.iOS18,
                compute_precision=ct.precision.FLOAT16,
                convert_to="mlprogram",
                skip_model_load=True,
            )
            part = Path(tmp) / f"b{b}.mlpackage"
            ml.save(str(part))
            del ml, traced, example
            gc.collect()
            if desc is None:
                desc = ct.utils.MultiFunctionDescriptor()
            desc.add_function(str(part), src_function_name="main", target_function_name=f"b{b}")
        desc.default_function_name = f"b{batches[0]}"
        if out.exists():
            shutil.rmtree(out)
        ct.utils.save_multifunction(desc, str(out))
    print(out)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("reference")
    for n in ("upstream", "samples_dir", "dataset_dir", "out"):
        r.add_argument(n)
    i = sub.add_parser("iree")
    i.add_argument("upstream")
    i.add_argument("out")
    i.add_argument("--batch", type=int, default=1)
    i.add_argument("--res", type=int, default=224)
    i.add_argument("--attention", choices=["sdpa", "decomposed"], default="sdpa")
    c = sub.add_parser("coreml")
    c.add_argument("upstream")
    c.add_argument("out")
    c.add_argument("--batches", default="1,8,32")
    c.add_argument("--res", type=int, default=224)
    a = p.parse_args()
    {"reference": cmd_reference, "iree": cmd_iree, "coreml": cmd_coreml}[a.cmd](a)


if __name__ == "__main__":
    main()
