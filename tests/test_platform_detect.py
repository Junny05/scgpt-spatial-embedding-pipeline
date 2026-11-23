"""Checkpoint-free tests for platform detection, gene-symbol column
selection/dedup, and auto min_genes, using small synthetic AnnData."""
import os
import sys

import numpy as np
import pandas as pd
import anndata as ad
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from platform_detect import (  # noqa: E402
    ConfigError,
    detect_platform,
    apply_in_tissue_filter,
    resolve_min_genes,
    select_gene_col_and_dedupe,
)


class FakeVocab:
    def __init__(self, tokens):
        self._tokens = set(tokens)

    def get_itos(self):
        return list(self._tokens)


DEFAULT_CFG = {"platform": None, "gene_col": None, "min_vocab_coverage": 0.90}


def make_visium(n_obs=20, n_vars=3000, with_in_tissue=True):
    X = np.random.poisson(1.0, size=(n_obs, n_vars)).astype(np.float32)
    var = pd.DataFrame(index=[f"gene{i}" for i in range(n_vars)])
    obs = pd.DataFrame(index=[f"spot{i}" for i in range(n_obs)])
    if with_in_tissue:
        obs["in_tissue"] = 1
    a = ad.AnnData(X=X, obs=obs, var=var)
    a.obsm["spatial"] = np.random.rand(n_obs, 2)
    return a


def make_targeted_spatial(n_obs=20, n_vars=300):
    X = np.random.poisson(1.0, size=(n_obs, n_vars)).astype(np.float32)
    var = pd.DataFrame(index=[f"gene{i}" for i in range(n_vars)])
    obs = pd.DataFrame(index=[f"cell{i}" for i in range(n_obs)])
    a = ad.AnnData(X=X, obs=obs, var=var)
    a.obsm["spatial"] = np.random.rand(n_obs, 2)
    return a


def make_snrna(n_obs=20, n_vars=50):
    X = np.random.poisson(1.0, size=(n_obs, n_vars)).astype(np.float32)
    ensembl_ids = [f"ENSG{i:05d}" for i in range(n_vars)]
    symbols = [f"GENE{i}" for i in range(n_vars)]
    var = pd.DataFrame({"gene_name": symbols}, index=ensembl_ids)
    obs = pd.DataFrame(index=[f"cell{i}" for i in range(n_obs)])
    return ad.AnnData(X=X, obs=obs, var=var)


def test_detect_visium_with_in_tissue():
    a = make_visium(with_in_tissue=True)
    is_spatial, platform = detect_platform(a, DEFAULT_CFG)
    assert is_spatial
    assert platform == "visium"


def test_detect_visium_hd_without_in_tissue():
    a = make_visium(with_in_tissue=False)
    is_spatial, platform = detect_platform(a, DEFAULT_CFG)
    assert is_spatial
    assert platform == "visium_hd"


def test_detect_targeted_panel_spatial():
    a = make_targeted_spatial()
    is_spatial, platform = detect_platform(a, DEFAULT_CFG)
    assert is_spatial
    assert platform == "xenium"  # xenium/merfish indistinguishable from data alone


def test_detect_snrna():
    a = make_snrna()
    is_spatial, platform = detect_platform(a, DEFAULT_CFG)
    assert not is_spatial
    assert platform == "snrna"


def test_platform_override_mismatch_raises():
    a = make_snrna()
    with pytest.raises(ConfigError):
        detect_platform(a, {**DEFAULT_CFG, "platform": "visium"})


def test_platform_override_accepted():
    a = make_snrna()
    is_spatial, platform = detect_platform(a, {**DEFAULT_CFG, "platform": "snrna"})
    assert platform == "snrna"


def test_in_tissue_filter_applied_when_present():
    a = make_visium(n_obs=10, with_in_tissue=True)
    a.obs.loc[a.obs_names[:4], "in_tissue"] = 0
    out = apply_in_tissue_filter(a)
    assert out.n_obs == 6


def test_in_tissue_filter_noop_when_absent():
    a = make_visium(n_obs=10, with_in_tissue=False)
    out = apply_in_tissue_filter(a)
    assert out.n_obs == 10


def test_resolve_min_genes_auto_whole_transcriptome():
    assert resolve_min_genes("auto", "visium", n_vars=20000) == 200


def test_resolve_min_genes_auto_targeted():
    assert resolve_min_genes("auto", "xenium", n_vars=300) == max(5, round(0.05 * 300))


def test_resolve_min_genes_auto_targeted_floor():
    assert resolve_min_genes("auto", "merfish", n_vars=50) == 5


def test_resolve_min_genes_explicit_override():
    assert resolve_min_genes(100, "xenium", n_vars=300) == 100


def test_gene_col_auto_selects_best_overlap_and_dedupes():
    n_vars = 6
    X = np.zeros((3, n_vars), dtype=np.float32)
    # make gene1 (dup of GENE_A at higher expression) win the dedupe
    X[:, 0] = 10  # GENE_A copy with low counts
    X[:, 1] = 50  # GENE_A copy with high counts -> should be kept
    X[:, 2] = 5   # GENE_B
    X[:, 3] = 1
    X[:, 4] = 1
    X[:, 5] = 1
    var = pd.DataFrame(
        {
            "gene_name": ["GENE_A", "GENE_A", "GENE_B", "UNMATCHED1", "UNMATCHED2", "UNMATCHED3"],
        },
        index=[f"ensg{i}" for i in range(n_vars)],
    )
    obs = pd.DataFrame(index=[f"c{i}" for i in range(3)])
    a = ad.AnnData(X=X, obs=obs, var=var)

    vocab = FakeVocab(["GENE_A", "GENE_B"])
    cfg = {**DEFAULT_CFG, "min_vocab_coverage": 0.30}
    out, chosen = select_gene_col_and_dedupe(a, vocab, cfg, lib_name="test")

    assert chosen == "gene_name"
    assert sorted(out.var["feature_name"]) == [
        "GENE_A", "GENE_B", "UNMATCHED1", "UNMATCHED2", "UNMATCHED3",
    ]
    # the higher-count GENE_A row (ensg1, X col 1) should be the one kept
    kept_ensg_a = out.var_names[out.var["feature_name"] == "GENE_A"][0]
    assert kept_ensg_a == "ensg1"


def test_gene_col_override():
    n_vars = 2
    X = np.ones((3, n_vars), dtype=np.float32)
    var = pd.DataFrame(
        {"gene_name": ["GENE_A", "GENE_B"], "gene_symbols": ["WRONG_A", "WRONG_B"]},
        index=["g0", "g1"],
    )
    obs = pd.DataFrame(index=[f"c{i}" for i in range(3)])
    a = ad.AnnData(X=X, obs=obs, var=var)
    vocab = FakeVocab(["GENE_A", "GENE_B", "WRONG_A", "WRONG_B"])
    out, chosen = select_gene_col_and_dedupe(
        a, vocab, {**DEFAULT_CFG, "gene_col": "gene_symbols"}, lib_name="test"
    )
    assert chosen == "gene_symbols"
    assert sorted(out.var["feature_name"]) == ["WRONG_A", "WRONG_B"]


def test_gene_col_low_overlap_raises():
    n_vars = 3
    X = np.ones((3, n_vars), dtype=np.float32)
    var = pd.DataFrame({"gene_name": ["NOPE1", "NOPE2", "NOPE3"]}, index=["g0", "g1", "g2"])
    obs = pd.DataFrame(index=[f"c{i}" for i in range(3)])
    a = ad.AnnData(X=X, obs=obs, var=var)
    vocab = FakeVocab(["SOMETHING_ELSE"])
    with pytest.raises(ConfigError):
        select_gene_col_and_dedupe(a, vocab, DEFAULT_CFG, lib_name="test")
