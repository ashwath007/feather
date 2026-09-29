"""A mismatched query vector must be rejected, not read past.

Every distance kernel reads exactly `dim` floats from the query pointer and
trusts the caller for the length. A SHORT query is therefore an out-of-bounds
read: measured before the fix, a 256-dim query against a 512-dim index read 1 KB
past the end of the buffer and returned a confident-looking score of 0.0076. A
LONG query is merely wrong — it silently compares a truncated prefix.

The Cloud API guarded this at the HTTP edge. Anyone using Feather embedded —
which is the entire point of the product — had no guard at all.
"""
import numpy as np
import pytest

import feather_db
from feather_db import DB

DIM = 512


@pytest.fixture
def db(tmp_path):
    d = DB.open(str(tmp_path / "g.feather"), dim=DIM)
    for i in range(5):
        m = feather_db.Metadata(); m.content = f"doc {i} alpha"
        d.add(i, np.random.rand(DIM).astype(np.float32), m)
    return d


def _q(n):
    return np.random.rand(n).astype(np.float32)


@pytest.mark.parametrize("n", [1, 128, 256, 511])
def test_short_query_is_rejected_not_read_past(db, n):
    """The memory-unsafe case."""
    with pytest.raises(ValueError, match="dimensions"):
        db.search(_q(n), k=1)


@pytest.mark.parametrize("n", [513, 768, 3072])
def test_long_query_is_rejected_not_truncated(db, n):
    with pytest.raises(ValueError, match="dimensions"):
        db.search(_q(n), k=1)


def test_correct_dim_still_works(db):
    assert len(db.search(_q(DIM), k=3)) == 3


def test_every_query_entry_point_is_guarded(db):
    """search is not the only path that takes a raw query vector."""
    with pytest.raises(ValueError):
        db.hybrid_search(_q(256), "alpha", k=1)
    with pytest.raises(ValueError):
        db.context_chain(_q(256), 2, 1, "text")


def test_the_error_names_both_dimensions(db):
    """A caller should not have to guess which side is wrong."""
    with pytest.raises(ValueError) as e:
        db.search(_q(256), k=1)
    msg = str(e.value)
    assert "256" in msg and "512" in msg and "text" in msg


def test_a_second_modality_is_checked_against_its_own_dim(tmp_path):
    """Multimodal pockets have independent dims; the guard must use the right one."""
    d = DB.open(str(tmp_path / "m.feather"), dim=768)
    m = feather_db.Metadata(); m.content = "x"
    d.add(1, np.random.rand(768).astype(np.float32), m)
    d.add(1, np.random.rand(512).astype(np.float32), m, modality="visual")

    assert len(d.search(_q(768), k=1)) == 1                    # text is 768
    assert len(d.search(_q(512), k=1, modality="visual")) == 1  # visual is 512
    with pytest.raises(ValueError):
        d.search(_q(768), k=1, modality="visual")              # text query, visual index
