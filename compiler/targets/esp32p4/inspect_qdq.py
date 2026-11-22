"""Inspect a QDQ ONNX: Q/DQ counts, dtypes, zero points, power-of-two scales,
per-channel axes, which float ops are left without QDQ, and ORT runnability."""

import collections
import math
import sys

import numpy as np
import onnx
from onnx import numpy_helper

path = sys.argv[1]
m = onnx.load(path)
g = m.graph
inits = {i.name: numpy_helper.to_array(i) for i in g.initializer}
consts = {}
for n in g.node:
    if n.op_type == "Constant":
        for a in n.attribute:
            if a.name == "value":
                consts[n.output[0]] = numpy_helper.to_array(a.t)
vals = {**inits, **consts}

print("opset", [(o.domain, o.version) for o in m.opset_import], "ir", m.ir_version)
print("ops", dict(collections.Counter(n.op_type for n in g.node)))

stats = collections.Counter()
non_pow2 = []
for n in g.node:
    if n.op_type not in ("QuantizeLinear", "DequantizeLinear"):
        continue
    scale = vals.get(n.input[1])
    zp = vals.get(n.input[2]) if len(n.input) > 2 else None
    axis = [a.i for a in n.attribute if a.name == "axis"]
    stats[(n.op_type, str(zp.dtype) if zp is not None else "none",
           "per-ch" if scale is not None and scale.size > 1 else "per-t")] += 1
    if zp is not None and np.any(zp != 0):
        stats["nonzero_zp"] += 1
    if scale is not None:
        e = np.log2(scale.astype(np.float64))
        if not np.allclose(e, np.round(e)):
            non_pow2.append((n.name, scale.flatten()[:4]))
for k, v in sorted(stats.items(), key=str):
    print(" ", k, v)
print("non-pow2 scales:", len(non_pow2), non_pow2[:5])

# Float compute ops whose inputs are not fed by DequantizeLinear.
producer = {o: n for n in g.node for o in n.output}
skip = {"QuantizeLinear", "DequantizeLinear", "Constant"}
unq = collections.Counter()
for n in g.node:
    if n.op_type in skip:
        continue
    srcs = [producer.get(i) for i in n.input if i]
    if not any(s is not None and s.op_type == "DequantizeLinear" for s in srcs):
        unq[n.op_type] += 1
print("ops with no DQ input:", dict(unq))

try:
    import onnxruntime as ort

    s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    i = s.get_inputs()[0]
    x = np.random.default_rng(0).uniform(-1, 1, i.shape).astype(np.float32)
    y = s.run(None, {i.name: x})[0]
    print("ORT ok", y.shape, float(y.min()), float(y.max()))
    if len(sys.argv) > 2:
        f = ort.InferenceSession(sys.argv[2], providers=["CPUExecutionProvider"])
        yf = f.run(None, {f.get_inputs()[0].name: x})[0]
        print("vs fp32: max abs diff", float(np.abs(y - yf).max()))
except Exception as e:  # noqa: BLE001
    print("ORT failed:", type(e).__name__, str(e)[:300])
