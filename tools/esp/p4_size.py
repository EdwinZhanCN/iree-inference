"""Code-size report for an IREE static-library object (`--iree-llvmcpu-link-static`):
section sizes, undefined symbols (resolved by the firmware link), and the
largest dispatch functions grouped by kind.

usage: p4_size.py <model.o> [top_n]
"""

import collections
import sys

from elftools.elf.elffile import ELFFile


def section_kind(name):
    if name.startswith(".debug"):
        return "debug"
    if name.startswith(".text"):
        return "text"
    if name.startswith((".rodata", ".srodata")):
        return "rodata"
    if name.startswith((".data", ".sdata", ".bss", ".sbss")):
        return "data"
    return "other"


def dispatch_kind(name):
    if "matmul" in name or "mmt4d" in name or "matvec" in name:
        return "matmul-like"
    if "conv" in name:
        return "conv"
    return "other"


def main():
    path = sys.argv[1]
    top_n = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    with open(path, "rb") as fh:
        elf = ELFFile(fh)
        sizes = collections.Counter()
        for s in elf.iter_sections():
            if s["sh_type"] not in ("SHT_RELA", "SHT_REL"):
                sizes[section_kind(s.name)] += s["sh_size"]
        symtab = elf.get_section_by_name(".symtab")
        undefined = sorted({s.name for s in symtab.iter_symbols() if s["st_shndx"] == "SHN_UNDEF" and s.name})
        funcs = sorted(
            ((s["st_size"], s.name) for s in symtab.iter_symbols() if s["st_info"]["type"] == "STT_FUNC" and s["st_size"]),
            reverse=True,
        )

    print("sections:", ", ".join(f"{k} {v / 1024:,.0f} KiB" for k, v in sizes.most_common()))
    print("undefined (firmware must provide):", ", ".join(undefined) or "none")
    total = sum(sz for sz, _ in funcs)
    by_kind = collections.Counter()
    for sz, name in funcs:
        by_kind[dispatch_kind(name)] += sz
    print(f"functions: {len(funcs)}, {total / 1024:,.0f} KiB")
    for kind, sz in by_kind.most_common():
        print(f"  {kind:12s} {sz / 1024:9,.0f} KiB  {sz / total * 100:5.1f}%")
    print(f"largest {top_n}:")
    for sz, name in funcs[:top_n]:
        print(f"  {sz / 1024:7,.0f} KiB  {name.rsplit('dispatch_', 1)[-1]}")


if __name__ == "__main__":
    main()
