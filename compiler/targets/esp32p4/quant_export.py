"""Quantize a PP-OCRv6 ONNX model with ESP-PPQ (esp32p4, w8a8) and export it
as .espdl and as QDQ ONNX (ESP-PPQ's ONNXRUNTIME exporter, inherited from
upstream PPQ).

Also saves ESP-PPQ's own quantized simulation (which models esp-dl arithmetic)
on the synthetic test images that compare_iree_ort.py uses.

usage: quant_export.py <src.onnx> <out_dir> <det|rec> N C H W

Outputs in <out_dir>: <kind>.fp32.onnx (fixed shape), <kind>.espdl,
<kind>.qdq.onnx, <kind>.qdq.json, <kind>.ppq_sim.npy.
"""

import os
import sys

import numpy as np
import onnx
import torch
from esp_ppq.api import espdl_quantize_onnx, export_ppq_graph
from esp_ppq.api.setting import QuantizationSettingFactory
from esp_ppq.core import TargetPlatform
from esp_ppq.executor import TorchExecutor
from onnx import helper, numpy_helper, shape_inference
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import synth  # noqa: E402

# Percentile calibration on 256 lines: on synthetic rec lines it gave 88.5%
# exact lines (fp32: 91%) against KL (esp-ppq's default) 77.5% and MSE 73%;
# 512 lines gave no more. Espressif's released model is likely calibrated on
# real crops; with real data, re-check this choice.
CALIB_IMAGES = 256
CALIB_ALGORITHM = "percentile"


def fix_input_shape(model, shape):
    dims = model.graph.input[0].type.tensor_type.shape.dim
    for d, v in zip(dims, shape):
        d.Clear()
        d.dim_value = v
    for o in model.graph.output:
        o.type.tensor_type.shape.Clear()
    return shape_inference.infer_shapes(model)


def _axes(n, vals):
    if len(n.input) > 1:
        return [int(a) for a in np.atleast_1d(vals.get(n.input[1], []))]
    return [a for at in n.attribute if at.name == "axes" for a in at.ints]


def conv1d_from_conv2d(model):
    """Paddle2ONNX writes Conv1D as Unsqueeze(2) -> Conv 1xk -> Squeeze(2).
    Rewrite it as one Conv over the 3D (N, C, L) tensor, as in Espressif's
    released model: no 4D detour, and a following BatchNormalization stays
    next to its Conv, where esp-ppq folds it in."""
    g = model.graph
    vals = _consts(g)
    inits = {i.name: i for i in g.initializer}
    prod = {o: n for n in g.node for o in n.output}
    users = {}
    for n in g.node:
        for i in n.input:
            users.setdefault(i, []).append(n)

    count = 0
    for conv in [n for n in g.node if n.op_type == "Conv"]:
        unsq = prod.get(conv.input[0])
        sqs = users.get(conv.output[0], [])
        w = inits.get(conv.input[1])
        if not (unsq is not None and unsq.op_type == "Unsqueeze" and _axes(unsq, vals) == [2]
                and len(users[unsq.output[0]]) == 1
                and len(sqs) == 1 and sqs[0].op_type == "Squeeze" and _axes(sqs[0], vals) == [2]
                and w is not None and len(w.dims) == 4 and w.dims[2] == 1):
            continue
        sq = sqs[0]
        attrs = {a.name: helper.get_attribute_value(a) for a in conv.attribute}
        if attrs.get("auto_pad", b"NOTSET") not in (b"NOTSET", "NOTSET"):
            continue
        k = list(attrs.get("kernel_shape", w.dims[2:]))
        st = list(attrs.get("strides", [1, 1]))
        dl = list(attrs.get("dilations", [1, 1]))
        pads = list(attrs.get("pads", [0, 0, 0, 0]))
        if k[0] != 1 or st[0] != 1 or dl[0] != 1 or pads[0] or pads[2]:
            continue
        w1 = numpy_helper.to_array(w)[:, :, 0, :]
        w1_name = f"{conv.input[1]}__1d"
        g.initializer.append(numpy_helper.from_array(w1, w1_name))
        node = helper.make_node(
            "Conv", [unsq.input[0], w1_name, *conv.input[2:]], [sq.output[0]],
            name=f"{conv.name or sq.output[0]}__1d", group=attrs.get("group", 1),
            kernel_shape=k[1:], strides=st[1:], dilations=dl[1:], pads=[pads[1], pads[3]])
        _replace(g, {unsq.output[0], conv.output[0], sq.output[0]}, [node])
        count += 1
    used = {i for n in g.node for i in n.input}
    for t in [t for t in g.initializer if t.name not in used]:
        g.initializer.remove(t)
    print(f"rewrote {count} Conv1D (Unsqueeze/Conv/Squeeze) as 1D Conv")
    del g.value_info[:]
    return shape_inference.infer_shapes(model)


def wrap_3d_batchnorm(model):
    """esp-ppq rewrites a BatchNormalization it cannot fold into a Conv as a 2D
    depthwise Conv, which mis-handles 3D (N, C, L) inputs. Wrap those as
    Unsqueeze(2) -> BN -> Squeeze(2); the result is unchanged. A 3D BN right
    after its Conv (see conv1d_from_conv2d) is folded and left alone."""
    g = model.graph
    ranks = {v.name: len(v.type.tensor_type.shape.dim) for v in list(g.value_info) + list(g.input)}
    opset = next(o.version for o in model.opset_import if o.domain in ("", "ai.onnx"))
    prod = {o: n for n in g.node for o in n.output}
    n_users = {}
    for n in g.node:
        for i in n.input:
            n_users[i] = n_users.get(i, 0) + 1
    nodes, wrapped = [], 0
    for n in g.node:
        src = prod.get(n.input[0])
        if (n.op_type != "BatchNormalization" or ranks.get(n.input[0]) != 3
                or (src is not None and src.op_type == "Conv" and n_users[n.input[0]] == 1)):
            nodes.append(n)
            continue
        x, y = n.input[0], n.output[0]
        x4, y4 = f"{n.name}__x4", f"{n.name}__y4"
        if opset >= 13:
            ax = f"{n.name}__ax"
            g.initializer.append(numpy_helper.from_array(np.array([2], dtype=np.int64), ax))
            unsq = helper.make_node("Unsqueeze", [x, ax], [x4])
            sq = helper.make_node("Squeeze", [y4, ax], [y])
        else:
            unsq = helper.make_node("Unsqueeze", [x], [x4], axes=[2])
            sq = helper.make_node("Squeeze", [y4], [y], axes=[2])
        n.input[0], n.output[0] = x4, y4
        nodes += [unsq, n, sq]
        wrapped += 1
    del g.node[:]
    g.node.extend(nodes)
    if wrapped:
        print(f"wrapped {wrapped} 3D BatchNormalization in Unsqueeze/Squeeze")
    return shape_inference.infer_shapes(model)


def _consts(g):
    vals = {i.name: numpy_helper.to_array(i) for i in g.initializer}
    for n in g.node:
        if n.op_type == "Constant" and n.attribute and n.attribute[0].name == "value":
            vals[n.output[0]] = numpy_helper.to_array(n.attribute[0].t)
    return vals


def _replace(g, drop, new):
    """Replace the nodes whose first output is in `drop` by `new`, in place."""
    first = min(i for i, n in enumerate(g.node) if n.output[0] in drop)
    keep = [n for n in g.node if n.output[0] not in drop]
    del g.node[:]
    g.node.extend(keep[:first] + new + keep[first:])


def strip_output_softmax(model):
    """Drop a Softmax that produces a graph output: the output becomes logits,
    as in Espressif's released model (CTC decoding only needs the argmax)."""
    g = model.graph
    outs = {o.name for o in g.output}
    for n in list(g.node):
        if n.op_type == "Softmax" and n.output[0] in outs:
            src, dst = n.input[0], n.output[0]
            g.node.remove(n)
            for m in g.node:
                m.output[:] = [dst if o == src else o for o in m.output]
                m.input[:] = [dst if i == src else i for i in m.input]
            print(f"removed output Softmax ({dst} is now logits)")
    return model


def fuse_activations(model):
    """Fuse the decomposed GELU and HardSwish that Paddle2ONNX writes at
    opset 11 into single Gelu / HardSwish nodes. esp-ppq then quantizes them as
    one int8 LUT each, like Espressif's released model; decomposed, the Erf is
    left in float. The result is only for esp-ppq: these ops need opset 20 / 14,
    and the rest of the graph is opset 11."""
    g = model.graph
    vals = _consts(g)
    prod = {o: n for n in g.node for o in n.output}
    users = {}
    for n in g.node:
        for i in n.input:
            users.setdefault(i, []).append(n)

    def const(name, value):
        v = vals.get(name)
        return v is not None and v.size == 1 and abs(float(v) - value) < 1e-5

    def only_user(name, op):
        u = users.get(name, [])
        return u[0] if len(u) == 1 and u[0].op_type == op else None

    def other(n, x):
        return n.input[1] if n.input[0] == x else n.input[0]

    gelu = hswish = 0
    for erf in [n for n in g.node if n.op_type == "Erf"]:
        # x / sqrt(2) -> Erf -> + 1 -> * x -> * 0.5
        div = prod.get(erf.input[0])
        if div is None or div.op_type != "Div" or not const(div.input[1], 2 ** 0.5):
            continue
        x = div.input[0]
        add = only_user(erf.output[0], "Add")
        mul1 = add and only_user(add.output[0], "Mul")
        mul2 = mul1 and only_user(mul1.output[0], "Mul")
        if not (add and const(other(add, erf.output[0]), 1.0)
                and mul1 and other(mul1, add.output[0]) == x
                and mul2 and const(other(mul2, mul1.output[0]), 0.5)):
            continue
        node = helper.make_node("Gelu", [x], [mul2.output[0]], name=f"{mul2.output[0]}__gelu")
        _replace(g, {n.output[0] for n in (div, erf, add, mul1, mul2)}, [node])
        gelu += 1
    for hs in [n for n in g.node if n.op_type == "HardSigmoid"]:
        # x * HardSigmoid(x; alpha = 1/6, beta = 1/2)
        attrs = {a.name: a.f for a in hs.attribute}
        mul = only_user(hs.output[0], "Mul")
        if not (mul and abs(attrs.get("alpha", 0.2) - 1 / 6) < 1e-6
                and abs(attrs.get("beta", 0.5) - 0.5) < 1e-6
                and other(mul, hs.output[0]) == hs.input[0]):
            continue
        node = helper.make_node("HardSwish", [hs.input[0]], [mul.output[0]],
                                name=f"{mul.output[0]}__hardswish")
        _replace(g, {hs.output[0], mul.output[0]}, [node])
        hswish += 1
    print(f"fused {gelu} Gelu, {hswish} HardSwish")
    return model


class SynthImages(Dataset):
    """Synthetic text images: realistic activation ranges, not an accuracy benchmark."""

    def __init__(self, kind, h, w, n):
        self.data = [torch.from_numpy(a[0]) for a in synth.batch(kind, n, h, w, seed=synth.CALIB_SEED)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


def main():
    src, out_dir, kind = sys.argv[1], sys.argv[2], sys.argv[3]
    shape = [int(v) for v in sys.argv[4:8]]
    os.makedirs(out_dir, exist_ok=True)

    model = fix_input_shape(onnx.load(src), shape)
    model = strip_output_softmax(wrap_3d_batchnorm(conv1d_from_conv2d(model)))
    fp32_path = os.path.join(out_dir, f"{kind}.fp32.onnx")
    onnx.save(model, fp32_path)
    # espdl_quantize_onnx overwrites its input with the simplified model.
    work_path = os.path.join(out_dir, f"{kind}.ppq_in.onnx")
    onnx.save(fuse_activations(model), work_path)

    loader = DataLoader(SynthImages(kind, shape[2], shape[3], CALIB_IMAGES), batch_size=1, shuffle=False)
    setting = QuantizationSettingFactory.espdl_setting(num_of_bits=8)
    setting.quantize_activation_setting.calib_algorithm = CALIB_ALGORITHM
    graph = espdl_quantize_onnx(
        onnx_import_file=work_path,
        espdl_export_file=os.path.join(out_dir, f"{kind}.espdl"),
        calib_dataloader=loader,
        calib_steps=CALIB_IMAGES,
        input_shape=[shape],
        target="esp32p4",
        quant_type="w8a8",
        setting=setting,
        device="cpu",
        error_report=False,
        verbose=0,
    )

    qdq_path = os.path.join(out_dir, f"{kind}.qdq.onnx")
    export_ppq_graph(
        graph=graph,
        platform=TargetPlatform.ONNXRUNTIME,
        graph_save_to=qdq_path,
        config_save_to=os.path.join(out_dir, f"{kind}.qdq.json"),
    )
    print("exported", qdq_path)

    executor = TorchExecutor(graph=graph, device="cpu")
    outs = [
        executor.forward(inputs=torch.from_numpy(x))[0].detach().cpu().numpy()
        for x in synth.batch(kind, synth.TEST_IMAGES, shape[2], shape[3], seed=synth.TEST_SEED)
    ]
    sim_path = os.path.join(out_dir, f"{kind}.ppq_sim.npy")
    np.save(sim_path, np.stack(outs))
    print("saved ESP-PPQ simulation", sim_path)


if __name__ == "__main__":
    main()
