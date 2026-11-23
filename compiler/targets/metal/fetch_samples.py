"""Download the iNaturalist sample photos listed in samples.json.

The photos are research-grade observations with CC0 or CC-BY licenses; each
entry carries the photographer attribution required by CC-BY. They are used
to check that exported BioCLIP 2 encoders agree with the fp32 reference on
embeddings and on top-k species from the TreeOfLife index.

usage: fetch_samples.py <out_dir>
"""

import json
import sys
import urllib.request
from pathlib import Path

MANIFEST = Path(__file__).with_name("samples.json")


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    for s in json.loads(MANIFEST.read_text()):
        path = out / f"{s['observation']}.jpg"
        if path.exists():
            continue
        req = urllib.request.Request(s["url"], headers={"User-Agent": "iree-inference-sample-fetch"})
        path.write_bytes(urllib.request.urlopen(req, timeout=60).read())
        print(f"{path.name}  {s['taxon']}  ({s['license']}) {s['attribution']}")


if __name__ == "__main__":
    main()
