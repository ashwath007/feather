"""auto_link's threshold is a cosine similarity, and must behave like one.

It used to be compared against `1/(1+L2_squared)`, which is not a similarity:
the documented default of 0.80 actually required cosine >= 0.875, and on
unnormalised embeddings it was unreachable. Measured before the fix, 20
documents at true cosine 0.9971 produced ZERO links, silently — and auto_link is
the primitive the clustering and entity-graph work sits on.
"""
import numpy as np
import pytest

import feather_db
from feather_db import DB

DIM = 256


def _corpus(db, n=12, sigma=0.05, scale=1.0, seed=5):
    """n vectors around one centroid. `scale` makes them deliberately non-unit."""
    rng = np.random.default_rng(seed)
    base = rng.normal(0, 1, DIM)
    vecs = []
    for i in range(n):
        v = ((base + rng.normal(0, sigma, DIM)) * scale).astype(np.float32)
        vecs.append(v)
        m = feather_db.Metadata(); m.content = f"doc {i}"
        db.add(i, v, m)
    return vecs


def _cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def test_links_are_created_on_unnormalised_vectors(tmp_path):
    """The regression: real embedders do not always return unit vectors."""
    db = DB.open(str(tmp_path / "u.feather"), dim=DIM)
    vecs = _corpus(db, scale=27.0)            # norms ~27, cosine ~0.997
    assert _cos(vecs[0], vecs[1]) > 0.95, "test corpus is not actually similar"
    assert np.linalg.norm(vecs[0]) > 5, "test corpus is unexpectedly unit-norm"

    assert db.auto_link("text", 0.80, "related_to") > 0, (
        "no links at threshold 0.80 on documents with cosine ~0.997")


def test_threshold_means_cosine(tmp_path):
    """Pairs above the threshold link; pairs below it do not."""
    db = DB.open(str(tmp_path / "t.feather"), dim=DIM)
    rng = np.random.default_rng(11)
    anchor = rng.normal(0, 1, DIM); anchor /= np.linalg.norm(anchor)

    # id 0 = anchor, id 1 ≈ cos 0.97, id 2 ≈ cos 0.50
    def at(target):
        p = rng.normal(0, 1, DIM); p -= np.dot(p, anchor) * anchor
        p /= np.linalg.norm(p)
        v = target * anchor + np.sqrt(max(0.0, 1 - target ** 2)) * p
        return (v / np.linalg.norm(v)).astype(np.float32)

    for i, v in enumerate([anchor.astype(np.float32), at(0.97), at(0.50)]):
        m = feather_db.Metadata(); m.content = f"v{i}"
        db.add(i, v, m)

    db.auto_link("text", 0.90, "related_to")
    targets = {e.target_id for e in db.get_edges(0)}
    assert 1 in targets, "cos 0.97 did not link at threshold 0.90"
    assert 2 not in targets, "cos 0.50 linked at threshold 0.90"


def test_a_high_threshold_links_less_than_a_low_one(tmp_path):
    """Monotonic in the threshold — the basic sanity property it lacked."""
    counts = {}
    for th in (0.99, 0.90, 0.50):
        db = DB.open(str(tmp_path / f"m{int(th*100)}.feather"), dim=DIM)
        _corpus(db, n=10, sigma=0.9, scale=3.0)
        counts[th] = db.auto_link("text", th, "related_to")
    assert counts[0.99] <= counts[0.90] <= counts[0.50], counts


def test_edge_weight_is_the_cosine(tmp_path):
    """The weight is persisted and flows into context_chain scoring, so it has
    to be the similarity it claims to be."""
    db = DB.open(str(tmp_path / "w.feather"), dim=DIM)
    vecs = _corpus(db, n=6, scale=9.0)
    db.auto_link("text", 0.80, "related_to")
    for e in db.get_edges(0):
        assert e.weight == pytest.approx(_cos(vecs[0], vecs[e.target_id]), abs=1e-3)
