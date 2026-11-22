"""Compare three runs of a quantized model on the same synthetic images:
ESP-PPQ's simulation (saved by quant_export.py), ONNX Runtime on the QDQ ONNX,
and IREE on the compiled vmfb. Also reports QDQ against fp32.

For rec, given PaddleOCR's inference.yml (its CTC dictionary), it also decodes
the text and measures line accuracy of fp32 and QDQ (ONNX Runtime; IREE is
compared to it above) on EVAL_LINES synthetic lines whose text is known.
Synthetic lines check the pipeline; they do not stand in for a real dataset.

With --plugin, IREE loads an executable plugin that resolves the esp-dl
kernel imports emitted by espdl_rewrite.py.

usage: compare_iree_ort.py [--plugin lib] <model.vmfb> <qdq.onnx> <fp32.onnx> <ppq_sim.npy> <det|rec> [inference.yml]
"""

import os
import sys

import numpy as np
import onnxruntime as ort
from iree import runtime as rt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import synth  # noqa: E402


def iree_entry(vmfb, plugin=None):
    if plugin:
        rt.flags.parse_flags(f"--executable_plugin={plugin}")
    ctx = rt.SystemContext(config=rt.Config("local-task"))
    with open(vmfb, "rb") as fh:
        ctx.add_vm_module(rt.VmModule.copy_buffer(ctx.instance, fh.read()))
    module = ctx.modules.module
    names = [n for n in module.vm_module.function_names if not n.startswith("__") and "$" not in n]
    fn = module[names[0]]

    def run(x):
        r = fn(x)
        return np.asarray(r.to_host() if hasattr(r, "to_host") else r)

    return run


EVAL_LINES = 200
EVAL_SEED = 4321


def ctc_charset(yml):
    import yaml

    chars = yaml.safe_load(open(yml, encoding="utf-8"))["PostProcess"]["character_dict"]
    return ["", *chars, " "]  # blank, dictionary, space (use_space_char)


def ctc_decode(logits, charset):
    ids = logits.reshape(-1, logits.shape[-1]).argmax(-1)
    keep = (ids != 0) & np.append(True, ids[1:] != ids[:-1])
    return "".join(charset[i] for i in ids[keep])


def line_accuracy(sessions, charset, shape):
    xs, texts = synth.rec_batch(EVAL_LINES, shape[2], shape[3], seed=EVAL_SEED)
    for name, s in sessions.items():
        out = [ctc_decode(s.run(None, {s.get_inputs()[0].name: x})[0], charset) for x in xs]
        exact = np.mean([o == t for o, t in zip(out, texts)]) * 100
        nospace = np.mean([o.replace(" ", "") == t.replace(" ", "") for o, t in zip(out, texts)]) * 100
        print(f"{name}: {exact:.1f}% lines exact, {nospace:.1f}% ignoring spaces ({EVAL_LINES} synthetic lines)")


def diff(a, b):
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return f"max|d| {d.max():.3e}, differing {(d > 0).mean() * 100:.4f}%"


def main():
    args = sys.argv[1:]
    plugin = None
    if args[:1] == ["--plugin"]:
        plugin, args = args[1], args[2:]
    vmfb, qdq_path, fp32_path, sim_path, kind = args[:5]
    charset = ctc_charset(args[5]) if kind == "rec" and len(args) > 5 else None
    q = ort.InferenceSession(qdq_path, providers=["CPUExecutionProvider"])
    f = ort.InferenceSession(fp32_path, providers=["CPUExecutionProvider"])
    run_iree = iree_entry(vmfb, plugin)
    sim = np.load(sim_path)
    shape = q.get_inputs()[0].shape

    if kind == "rec":
        images, texts = synth.rec_batch(synth.TEST_IMAGES, shape[2], shape[3], seed=synth.TEST_SEED)
    else:
        images = synth.batch(kind, synth.TEST_IMAGES, shape[2], shape[3], seed=synth.TEST_SEED)
    for t, x in enumerate(images):
        y_q = q.run(None, {q.get_inputs()[0].name: x})[0]
        y_f = f.run(None, {f.get_inputs()[0].name: x})[0]
        y_i = run_iree(x)
        print(f"[{t}] ESP-PPQ sim vs ORT-QDQ: {diff(sim[t], y_q)}")
        print(f"[{t}] IREE vs ORT-QDQ:        {diff(y_i, y_q)}")
        if kind == "rec":
            agree = lambda a, b: (a.argmax(-1) == b.argmax(-1)).mean() * 100  # noqa: E731
            print(
                f"[{t}] argmax agree: IREE/ORT {agree(y_i, y_q):.1f}%, IREE/sim {agree(y_i, sim[t]):.1f}%, "
                f"QDQ/fp32 {agree(y_q, y_f):.1f}%"
            )
            if charset:
                print(f"[{t}] text {texts[t]!r}: fp32 {ctc_decode(y_f, charset)!r}, "
                      f"QDQ {ctc_decode(y_q, charset)!r}, IREE {ctc_decode(y_i, charset)!r}")
        else:
            print(f"[{t}] text px (>0.3): fp32 {(y_f > 0.3).mean() * 100:.2f}%, QDQ {(y_q > 0.3).mean() * 100:.2f}%")
    if charset:
        line_accuracy({"fp32": f, "QDQ": q}, charset, shape)


if __name__ == "__main__":
    main()
