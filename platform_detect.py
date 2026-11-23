"""
Platform detection, gene-symbol column selection, and gene deduplication,
shared across Visium / Visium HD / Xenium / MERFISH / snRNA-seq inputs.

Design notes:
- spatial vs. snRNA is decided by obsm["spatial"] presence, not in_tissue
  (Visium HD commonly has no in_tissue column but is still spatial).
- in_tissue filtering is applied only if the column exists.
- platform label is a best-effort guess from data shape, always overridable
  via the `platform` config key, and only used for logging / min_genes
  defaults - it never gates which code path runs (spatial vs. non-spatial
  and the gene_col auto-selection are the load-bearing decisions).
"""
from __future__ import annotations

import numpy as np

WHOLE_TRANSCRIPTOME_CUTOFF = 2000  # n_vars above this => whole-transcriptome-like
GENE_COL_CANDIDATES = ["gene_name", "gene_symbols", "feature_name"]

VALID_PLATFORMS = {"visium", "visium_hd", "xenium", "merfish", "snrna"}


class ConfigError(ValueError):
    """Raised instead of a bare KeyError/IndexError so the message always
    names the config key that fixes the problem."""


def detect_platform(adata, cfg):
    """Return (is_spatial, platform) without mutating adata.

    Never raises a bare KeyError: any ambiguity is reported with the config
    key that resolves it.
    """
    is_spatial = "spatial" in adata.obsm
    has_in_tissue = "in_tissue" in adata.obs
    n_vars = adata.n_vars

    platform = cfg.get("platform")
    if platform is not None:
        if platform not in VALID_PLATFORMS:
            raise ConfigError(
                f"config['platform'] = {platform!r} is not one of {sorted(VALID_PLATFORMS)}."
            )
        platform_is_spatial = platform != "snrna"
        if platform_is_spatial != is_spatial:
            raise ConfigError(
                f"config['platform'] = {platform!r} implies "
                f"{'spatial' if platform_is_spatial else 'non-spatial'} data, but "
                f"obsm['spatial'] {'is' if is_spatial else 'is not'} present. "
                "Fix the 'platform' key or check the input file."
            )
        return is_spatial, platform

    if not is_spatial:
        return False, "snrna"

    is_targeted = n_vars <= WHOLE_TRANSCRIPTOME_CUTOFF
    if is_targeted:
        # Xenium and MERFISH both ship targeted panels with no in_tissue
        # column; we can't tell them apart from the AnnData alone.
        platform = "xenium"
    else:
        platform = "visium" if has_in_tissue else "visium_hd"
    return True, platform


def apply_in_tissue_filter(adata, lib_name=""):
    """Filter to obs['in_tissue'] == 1 only if that column exists."""
    if "in_tissue" in adata.obs:
        return adata[adata.obs["in_tissue"] == 1].copy()
    return adata


def resolve_min_genes(min_genes_cfg, platform, n_vars):
    """min_genes_cfg is "auto" or an explicit int (config key: min_genes)."""
    if min_genes_cfg != "auto":
        try:
            return int(min_genes_cfg)
        except (TypeError, ValueError):
            raise ConfigError(
                f"config['min_genes'] = {min_genes_cfg!r} must be 'auto' or an integer."
            )
    is_targeted = platform in ("xenium", "merfish")
    if is_targeted:
        return max(5, round(0.05 * n_vars))
    return 200


def _dedupe_by_symbol(adata, symbols):
    """Keep, for each duplicated symbol, the var row with the highest total
    counts (summed over the cells/spots currently in adata). `symbols` is a
    pandas Series aligned to adata.var_names holding the candidate gene
    symbol for each var row (may contain NaN/empty for genes to drop)."""
    import pandas as pd

    symbols = symbols.astype(str)
    valid = symbols.notna() & (symbols != "") & (symbols != "nan")
    adata = adata[:, valid.values].copy()
    symbols = symbols[valid.values]

    counts_per_gene = np.asarray(adata.X.sum(axis=0)).flatten()
    order = pd.Series(counts_per_gene, index=adata.var_names).sort_values(ascending=False).index
    symbol_by_var = symbols.reindex(order)
    dup = symbol_by_var.duplicated(keep="first")
    keep = set(order[~dup.values])

    keep_sorted = [v for v in adata.var_names if v in keep]
    adata = adata[:, keep_sorted].copy()
    adata.var["feature_name"] = symbols.reindex(adata.var_names).values
    return adata


def select_gene_col_and_dedupe(adata, vocab, cfg, lib_name=""):
    """Pick the var column holding gene symbols (best overlap with vocab,
    or the `gene_col` config override), dedupe by symbol, and return
    (adata_with_feature_name_col, chosen_gene_col)."""
    vocab_itos = set(vocab.get_itos())
    min_coverage = cfg["min_vocab_coverage"]

    gene_col_override = cfg.get("gene_col")
    candidates = {}
    for col in GENE_COL_CANDIDATES:
        if col in adata.var:
            candidates[col] = adata.var[col]
    candidates["var_names (index)"] = adata.var_names.to_series(index=adata.var_names)

    def overlap(series):
        vals = series.astype(str)
        vals = vals[vals.notna() & (vals != "") & (vals != "nan")]
        if len(vals) == 0:
            return 0.0
        return len(set(vals) & vocab_itos) / len(vals)

    scored = {name: overlap(series) for name, series in candidates.items()}
    for name, score in scored.items():
        print(f"[{lib_name}] gene_col candidate '{name}': vocab overlap={score:.1%}", flush=True)

    if gene_col_override is not None:
        if gene_col_override not in candidates:
            raise ConfigError(
                f"config['gene_col'] = {gene_col_override!r} is not a valid column "
                f"(tried: {list(candidates)}). Set 'gene_col' to one of those, or omit it "
                "for auto-selection."
            )
        chosen = gene_col_override
    else:
        chosen = max(scored, key=scored.get)
        if scored[chosen] < min_coverage:
            raise ConfigError(
                f"[{lib_name}] best gene_col candidate '{chosen}' only has "
                f"{scored[chosen]:.1%} overlap with the model vocabulary, below "
                f"min_vocab_coverage ({min_coverage:.1%}). Set 'gene_col' in config.yaml "
                f"to the correct var column, or lower 'min_vocab_coverage' if this is expected."
            )

    print(f"[{lib_name}] using gene_col='{chosen}' (overlap={scored[chosen]:.1%})", flush=True)
    symbols = candidates[chosen]
    return _dedupe_by_symbol(adata, symbols), chosen
