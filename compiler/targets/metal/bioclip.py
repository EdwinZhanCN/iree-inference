"""Shared BioCLIP 2 helpers: the upstream model, preprocessing, and the
TreeOfLife zero-shot index.

The index is the one Lumen-Hub ships (`datasets/TreeOfLife200MCore.*`): an
`[N, 768]` float16 matrix of L2-normalized text embeddings, a JSON list of
taxonomy records in the same order, and an hnswlib graph (`.bin`) that is not
used here. Classification follows Lumen-Hub's `dataset.rs`: rank by cosine
similarity, softmax with `exp(logit_scale) = 100`. This module ranks over the
whole matrix (exact); Lumen-Hub reranks 200 HNSW candidates, which gives the
same top-k when the HNSW recall is good.
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image

LOGIT_SCALE = 100.0
RANKS = ["kingdom", "phylum", "class", "order", "family", "genus", "species", "common_name"]


def load_model(upstream_dir):
    """The open_clip model in fp32 eval mode and its image transform."""
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(f"local-dir:{upstream_dir}")
    return model.eval(), preprocess


def load_images(paths, preprocess):
    """`[N, 3, 224, 224]` float32, preprocessed as open_clip does."""
    return np.stack([preprocess(Image.open(p).convert("RGB")).numpy() for p in paths])


class TreeOfLife:
    def __init__(self, dataset_dir, name="TreeOfLife200MCore"):
        dataset_dir = Path(dataset_dir)
        self.embeddings = np.load(dataset_dir / f"{name}.npy", mmap_mode="r")
        records = json.loads((dataset_dir / f"{name}.json").read_text())
        self.labels = ["/".join(r.get(k, "").strip() or "*" for k in RANKS) for r in records]
        assert len(self.labels) == self.embeddings.shape[0]

    def similarities(self, queries, chunk=65536):
        """Cosine similarity `[Q, N]` of float32 queries against every entry."""
        q = np.asarray(queries, np.float32)
        q = q / np.linalg.norm(q, axis=1, keepdims=True)
        out = np.empty((q.shape[0], self.embeddings.shape[0]), np.float32)
        for i in range(0, self.embeddings.shape[0], chunk):
            e = np.asarray(self.embeddings[i:i + chunk], np.float32)
            with np.errstate(divide="ignore", invalid="ignore"):
                out[:, i:i + chunk] = (q @ e.T) / np.linalg.norm(e, axis=1)
        # Zero rows (as in Lumen-Hub, NaN there) never rank.
        out[~np.isfinite(out)] = -np.inf
        return out

    def top_k(self, queries, k=5):
        """Per query, `[(index, cosine, softmax score), ...]` best first."""
        sims = self.similarities(queries)
        results = []
        for s in sims:
            idx = np.argpartition(-s, k)[:k]
            idx = idx[np.argsort(-s[idx])]
            z = np.exp((s - s.max()) * LOGIT_SCALE)
            results.append([(int(i), float(s[i]), float(z[i] / z.sum())) for i in idx])
        return results
