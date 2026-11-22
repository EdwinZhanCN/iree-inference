"""Test vectors for the esp-dl backend on ESP32-P4.

Takes the kernel configurations espdl_rewrite.py matches in the models
(convolution shapes, strides, padding, groups, ReLU, exponents; Add/Mul
broadcasts and exponents), shrinks spatial sizes, fills inputs and weights
with random int8 and computes the expected outputs with the reference kernels
(espdl_ref.c, via ctypes). Writes one C file for espdl_p4_test.

usage: gen_vectors.py <libespdl_ref> <out.c> <model.input.mlir>...
"""

import ctypes
import sys

import numpy as np
from iree.compiler import ir

import espdl_rewrite as R

RNG = np.random.default_rng(7)
MAX_H, MAX_W = 7, 12   # spatial crop of convolution cases
BIN_ROWS = 5           # rows of binary cases: n = rows * b_len (b_len = n for same shape)


def collect(paths):
    convs, binaries, pools = {}, {}, {}
    for path in paths:
        with ir.Context(), ir.Location.unknown():
            module = ir.Module.parse(open(path).read())
            for op in R.walk(module.operation):
                if op.name != "iree_linalg_ext.quantize_affine":
                    continue
                m = None
                for matcher in (R.match, R.match_pool, R.match_binary):
                    try:
                        m = matcher(op)
                        break
                    except R.Skip:
                        continue
                if m is None:
                    continue
                if m["kind"] == "conv":
                    key = tuple(m[k] for k in ("C", "O", "KH", "KW", "SH", "SW", "PT", "PL", "PB",
                                                "PR", "DH", "DW", "G", "RELU"))
                    if key not in convs:
                        convs[key] = {k: m[k] for k in R.PARAMS if k in m} | {
                            "w_exp": m["w_exp"].copy(), "H": m["H"], "W": m["W"]}
                elif m["kind"] == "pool":
                    key = (m["C"], m["KH"], m["KW"], m["SH"], m["SW"], m["X_EXP"], m["Y_EXP"])
                    if key not in pools:
                        pools[key] = {k: m[k] for k in R.POOL_PARAMS}
                else:
                    rank = len(m["a_shape"])
                    full = m["b_full"] == m["a_shape"]
                    bl = int(np.prod([d for d in m["b_full"] if d != 1])) if not full else 0
                    key = (m["op"], full, bl == 1, m["ea"], m["eb"], m["ey"])
                    if key not in binaries:
                        binaries[key] = {"op": m["op"], "ea": m["ea"], "eb": m["eb"], "ey": m["ey"],
                                         "full": full, "bl": bl, "rank": rank}
    return list(convs.values()), list(binaries.values()), list(pools.values())


def conv_case(c, lib):
    p = dict(c)
    p["H"], p["W"] = min(c["H"], MAX_H), min(c["W"], MAX_W)
    for dim, k, s, d, lo, hi in (("H", "KH", "SH", "DH", "PT", "PB"), ("W", "KW", "SW", "DW", "PL", "PR")):
        # Keep at least one full window.
        p[dim] = max(p[dim], (p[k] - 1) * p[d] + 1 - p[lo] - p[hi], 1)
    p["OH"] = (p["H"] + p["PT"] + p["PB"] - p["DH"] * (p["KH"] - 1) - 1) // p["SH"] + 1
    p["OW"] = (p["W"] + p["PL"] + p["PR"] - p["DW"] * (p["KW"] - 1) - 1) // p["SW"] + 1
    cpg = p["C"] // p["G"]
    x = RNG.integers(-128, 128, (p["H"], p["W"], p["C"]), dtype=np.int8)
    w = RNG.integers(-128, 128, (p["O"], p["KH"], p["KW"], cpg), dtype=np.int8)
    wp = R.pack_filter(w, p["G"] > 1)
    bias = RNG.integers(-(1 << 14), 1 << 14, p["O"], dtype=np.int32)
    w_exp = c["w_exp"].astype(np.int8)
    params = np.array([p[k] for k in R.PARAMS], dtype=np.int32)
    y = np.zeros(p["OH"] * p["OW"] * p["O"], dtype=np.int8)
    rc = lib.espdl_ref_conv2d_s8(ptr(x), ptr(wp), ptr(bias), ptr(w_exp), ptr(params), ptr(y))
    assert rc == 0, rc
    name = (f"conv C{p['C']} O{p['O']} {p['KH']}x{p['KW']} s{p['SH']}x{p['SW']} "
            f"p{p['PT']}{p['PL']}{p['PB']}{p['PR']} g{p['G']}{' relu' if p['RELU'] else ''} "
            f"{p['H']}x{p['W']}")
    return name, "conv", [x, wp, bias, w_exp, params], y


def binary_case(b, lib):
    if b["full"]:
        bl = RNG.integers(8, 64) * 1
        n = bl * BIN_ROWS
        bl = n
    else:
        bl = max(b["bl"], 1)
        n = bl * BIN_ROWS
    a = RNG.integers(-128, 128, n, dtype=np.int8)
    bb = RNG.integers(-128, 128, bl, dtype=np.int8)
    params = np.array([n, bl, {"add": 0, "mul": 1}[b["op"]], b["ea"], b["eb"], b["ey"]], dtype=np.int32)
    y = np.zeros(n, dtype=np.int8)
    rc = lib.espdl_ref_binary_s8(ptr(a), ptr(bb), ptr(params), ptr(y))
    assert rc == 0, rc
    name = f"{b['op']} n{n} b{bl} e{b['ea']},{b['eb']}->{b['ey']}"
    return name, "binary", [a, bb, params], y


def pool_case(p0, lib):
    p = dict(p0)
    # Keep global pools (KH == H, KW == W) global after shrinking large spatial dims.
    if p["KH"] == p["H"] and p["KW"] == p["W"]:
        p["H"] = p["KH"] = min(p["H"], MAX_H)
        p["W"] = p["KW"] = min(p["W"], 23 if p["W"] >= 23 and (p["W"] % 2) else MAX_W)
        p["OH"], p["OW"] = 1, 1
    else:
        p["H"] = max(min(p["H"], MAX_H), p["KH"])
        p["W"] = max(min(p["W"], MAX_W), p["KW"])
        p["OH"] = (p["H"] - p["KH"]) // p["SH"] + 1
        p["OW"] = (p["W"] - p["KW"]) // p["SW"] + 1
    x = RNG.integers(-128, 128, (p["H"], p["W"], p["C"]), dtype=np.int8)
    params = np.array([p[k] for k in R.POOL_PARAMS], dtype=np.int32)
    y = np.zeros(p["OH"] * p["OW"] * p["C"], dtype=np.int8)
    rc = lib.espdl_ref_avg_pool2d_s8(ptr(x), ptr(params), ptr(y))
    assert rc == 0, rc
    name = f"pool C{p['C']} {p['H']}x{p['W']} k{p['KH']}x{p['KW']} s{p['SH']}x{p['SW']}"
    return name, "pool", [x, params], y


def lut_case(n, lib):
    x = RNG.integers(-128, 128, n, dtype=np.int8)
    table = RNG.integers(-128, 128, 256, dtype=np.int8)
    params = np.array([n], dtype=np.int32)
    y = np.zeros(n, dtype=np.int8)
    assert lib.espdl_ref_lut_s8(ptr(x), ptr(table), ptr(params), ptr(y)) == 0
    return f"lut n{n}", "lut", [x, table, params], y


def ptr(a):
    return a.ctypes.data_as(ctypes.c_void_p)


def c_array(name, a):
    ctype = {np.dtype(np.int8): "int8_t", np.dtype(np.int32): "int32_t"}[a.dtype]
    vals = ",".join(str(int(v)) for v in a.reshape(-1))
    return f"static const {ctype} {name}[{max(a.size, 1)}] __attribute__((aligned(16))) = {{{vals}}};"


def main():
    lib = ctypes.CDLL(sys.argv[1])
    out, paths = sys.argv[2], sys.argv[3:]
    convs, binaries, pools = collect(paths)
    cases = [conv_case(c, lib) for c in convs]
    cases += [pool_case(p, lib) for p in pools]
    cases += [binary_case(b, lib) for b in binaries]
    cases += [lut_case(n, lib) for n in (1, 15, 16, 1000)]

    lines = ["// Generated by gen_vectors.py. Do not edit.",
             '#include "vectors.h"', ""]
    entries = []
    for i, (name, kind, ins, y) in enumerate(cases):
        names = []
        for j, a in enumerate(ins):
            lines.append(c_array(f"c{i}_in{j}", a))
            names.append(f"c{i}_in{j}")
        lines.append(c_array(f"c{i}_out", y))
        names += ["NULL"] * (5 - len(names))
        entries.append(f'  {{"{name}", ESPDL_CASE_{kind.upper()}, {{{", ".join(names)}}}, '
                       f"c{i}_out, {y.size}}},")
    lines += ["", "const espdl_case_t espdl_cases[] = {", *entries, "};",
              f"const int espdl_case_count = {len(cases)};", ""]
    with open(out, "w") as f:
        f.write("\n".join(lines))
    print(f"{len(convs)} conv, {len(pools)} pool, {len(binaries)} binary, 4 lut cases -> {out}")


if __name__ == "__main__":
    main()
