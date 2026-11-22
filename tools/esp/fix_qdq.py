"""Make esp-ppq's QDQ ONNX export standard opset 13, so ONNX Runtime and
iree-import-onnx accept it.

- InsertZeros (esp-ppq's private op for ConvTranspose) with stride s and
  output_padding p on NCHW:
    out[..., i*s, j*s] = x[..., i, j], zeros elsewhere,
    out spatial = (in - 1) * s + 1 + p   (p < s, as for ConvTranspose)
  becomes Unsqueeze -> Pad -> Reshape -> Slice.
- Gelu (opset 20) becomes 0.5 * x * (1 + Erf(x / sqrt(2))).
- HardSwish (opset 14) becomes x * HardSigmoid(x; 1/6, 1/2).

Gelu and HardSwish sit between a DequantizeLinear and a QuantizeLinear, so
each stays one int8 -> int8 elementwise op (an int8 LUT in esp-dl).

usage: fix_qdq.py <in.onnx> <out.onnx>
"""

import collections
import sys

import numpy as np
import onnx
from onnx import helper, numpy_helper, shape_inference

src, dst = sys.argv[1], sys.argv[2]
m = shape_inference.infer_shapes(onnx.load(src))
g = m.graph
shapes = {}
for vi in list(g.value_info) + list(g.input) + list(g.output):
    dims = [d.dim_value for d in vi.type.tensor_type.shape.dim]
    if dims and all(d > 0 for d in dims):
        shapes[vi.name] = dims


def const(name, arr, dtype=np.int64):
    g.initializer.append(numpy_helper.from_array(np.asarray(arr, dtype=dtype), name))
    return name


def insert_zeros(n, p):
    attrs = {a.name: helper.get_attribute_value(a) for a in n.attribute}
    sh, sw = attrs.get("stride", [1, 1])
    ph, pw = attrs.get("output_padding", [0, 0])
    assert ph < sh and pw < sw, (n.name, attrs)
    x, y = n.input[0], n.output[0]
    N, C, H, W = shapes[x]
    pads = [0, 0, 0, 0, 0, 0, 0, 0, 0, sh - 1, 0, sw - 1]
    end = [(H - 1) * sh + 1 + ph, (W - 1) * sw + 1 + pw]
    return [
        helper.make_node("Unsqueeze", [x, const(p + "_ax", [3, 5])], [p + "_u"]),
        helper.make_node("Pad", [p + "_u", const(p + "_pads", pads)], [p + "_p"]),
        helper.make_node("Reshape", [p + "_p", const(p + "_shape", [N, C, H * sh, W * sw])], [p + "_r"]),
        helper.make_node(
            "Slice",
            [p + "_r", const(p + "_st", [0, 0]), const(p + "_en", end), const(p + "_axes", [2, 3])],
            [y],
        ),
    ]


def gelu(n, p):
    x, y = n.input[0], n.output[0]
    f32 = np.float32
    return [
        helper.make_node("Div", [x, const(p + "_sqrt2", np.sqrt(2.0), f32)], [p + "_d"]),
        helper.make_node("Erf", [p + "_d"], [p + "_e"]),
        helper.make_node("Add", [p + "_e", const(p + "_one", 1.0, f32)], [p + "_a"]),
        helper.make_node("Mul", [x, p + "_a"], [p + "_m"]),
        helper.make_node("Mul", [p + "_m", const(p + "_half", 0.5, f32)], [y]),
    ]


def hard_swish(n, p):
    x, y = n.input[0], n.output[0]
    return [
        helper.make_node("HardSigmoid", [x], [p + "_hs"], alpha=1.0 / 6.0, beta=0.5),
        helper.make_node("Mul", [x, p + "_hs"], [y]),
    ]


LOWER = {"InsertZeros": insert_zeros, "Gelu": gelu, "HardSwish": hard_swish}

new_nodes, replaced = [], collections.Counter()
for i, n in enumerate(g.node):
    lower = LOWER.get(n.op_type) if n.domain in ("", "ai.onnx") else None
    if lower is None:
        new_nodes.append(n)
        continue
    new_nodes += lower(n, f"{n.name or n.output[0]}__{n.op_type.lower()}{i}")
    replaced[n.op_type] += 1

del g.node[:]
g.node.extend(new_nodes)
onnx.checker.check_model(m)
onnx.save(m, dst)
print(f"lowered {dict(replaced) or 'nothing'} -> {dst}")
