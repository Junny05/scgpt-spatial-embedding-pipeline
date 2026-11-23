"""Checkpoint-free correctness tests for GeneAggregator (min_count/n_obs)."""
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

# gene_embed imports scgpt_spatial, which lives under the vendored
# scGPT-spatial/ dir - normally embed_batch.py puts that on sys.path before
# importing gene_embed. Mirror that here so this test has no other
# dependency on config.yaml/a checkpoint being set up.
_vendored = os.path.join(REPO_ROOT, "scGPT-spatial")
if os.path.isdir(_vendored):
    sys.path.insert(0, _vendored)

from gene_embed import GeneAggregator  # noqa: E402


class FakeVocab:
    def __init__(self, itos):
        self._itos = itos

    def get_itos(self):
        return self._itos


def test_aggregator_averages_and_counts_correctly():
    vocab_size, embsize = 4, 2
    agg = GeneAggregator(vocab_size=vocab_size, embsize=embsize)

    # batch of 2 "cells", both in region "r1", gene 0 seen in both
    region_labels = ["r1", "r1"]
    gene_ids = np.array([[0, 1], [0, 2]])
    gene_embs = np.array(
        [[[1.0, 1.0], [2.0, 2.0]], [[3.0, 3.0], [4.0, 4.0]]], dtype=np.float32
    )
    mask = np.array([[True, True], [True, False]])  # second cell's gene 2 is padding
    agg.add(region_labels, gene_ids, gene_embs, mask)

    out = agg.finalize(FakeVocab(["g0", "g1", "g2", "g3"]), min_count=1)
    rows = {(r, g): (x, n) for r, g, x, n in zip(out.obs["region"], out.obs["gene"], out.X, out.obs["n_obs"])}

    assert ("r1", "g0") in rows
    x0, n0 = rows[("r1", "g0")]
    assert n0 == 2
    np.testing.assert_allclose(x0, [2.0, 2.0])  # mean of [1,1] and [3,3]

    x1, n1 = rows[("r1", "g1")]
    assert n1 == 1
    np.testing.assert_allclose(x1, [2.0, 2.0])

    # gene 2 was masked out (padding) in its only occurrence -> never added
    assert ("r1", "g2") not in rows


def test_aggregator_min_count_drops_rare_rows():
    agg = GeneAggregator(vocab_size=3, embsize=1)
    region_labels = ["r1"]
    gene_ids = np.array([[0]])
    gene_embs = np.array([[[5.0]]], dtype=np.float32)
    mask = np.array([[True]])
    agg.add(region_labels, gene_ids, gene_embs, mask)

    out_strict = agg.finalize(FakeVocab(["g0", "g1", "g2"]), min_count=2)
    assert out_strict.n_obs == 0

    out_loose = agg.finalize(FakeVocab(["g0", "g1", "g2"]), min_count=1)
    assert out_loose.n_obs == 1
    assert out_loose.obs["n_obs"].iloc[0] == 1


def test_aggregator_separates_regions():
    agg = GeneAggregator(vocab_size=2, embsize=1)
    agg.add(["a"], np.array([[0]]), np.array([[[1.0]]], dtype=np.float32), np.array([[True]]))
    agg.add(["b"], np.array([[0]]), np.array([[[9.0]]], dtype=np.float32), np.array([[True]]))

    out = agg.finalize(FakeVocab(["g0", "g1"]), min_count=1)
    regions = set(out.obs["region"])
    assert regions == {"a", "b"}
