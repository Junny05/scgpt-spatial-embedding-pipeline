import sys, os, time, csv, random
import numpy as np
import scanpy as sc
import torch

from config import load_config
from platform_detect import (
    detect_platform,
    apply_in_tissue_filter,
    resolve_min_genes,
    select_gene_col_and_dedupe,
)

CFG = load_config(os.environ.get("SCGPT_CONFIG", "config.yaml"))
REPO = CFG["repo"]
CKPT = CFG["checkpoint"]
DATA = CFG["data_dir"]
OUT = CFG["out_dir"]
LOG = CFG["log_file"]
MIN_GENES_CFG = CFG["min_genes"]
BATCH_SIZE = CFG["batch_size"]
MAX_LENGTH = CFG["max_length"]
LEIDEN_RES = CFG["leiden_resolution"]
N_SAMPLE_CELLS = CFG["n_sample_cells"]
GENE_LIST = set(CFG["gene_list"])
MIN_VOCAB_COVERAGE = CFG["min_vocab_coverage"]
MIN_SPOTS_PER_GENE_REGION = CFG["min_spots_per_gene_region"]
SEED = CFG["seed"]
MAX_UNITS_FOR_UMAP = CFG["max_units_for_umap"]
DEVICE_CFG = CFG["device"]

sys.path.insert(0, REPO)
from scgpt_spatial.tasks import embed_data  # noqa: E402

import gene_embed  # noqa: E402

OBS_COLS_TO_CARRY = [
    "clinical_group", "tissue_source", "subject_id", "region", "sample_id",
    "Batch", "Source", "individualID", "libraryID",
]


def resolve_device(device_cfg):
    if device_cfg == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_cfg


DEVICE = resolve_device(DEVICE_CFG)


def print_vocab_coverage(lib, adata, vocab, gene_col="feature_name"):
    genes = adata.var[gene_col].astype(str)
    in_vocab = genes.isin(vocab.get_itos())
    counts_per_gene = np.asarray(adata.X.sum(axis=0)).flatten()
    total_counts = counts_per_gene.sum()
    in_vocab_counts = counts_per_gene[in_vocab.values].sum()
    expressed_in_vocab = int(((counts_per_gene > 0) & in_vocab.values).sum())
    share = in_vocab_counts / total_counts if total_counts > 0 else float("nan")

    print(
        f"[{lib}] vocab coverage: genes_in_vocab={int(in_vocab.sum())}/{len(genes)} | "
        f"expressed_genes_in_vocab={expressed_in_vocab} | "
        f"share_of_counts_on_in_vocab_genes={share:.4f}",
        flush=True,
    )
    if share < MIN_VOCAB_COVERAGE:
        print(
            f"[WARNING] {lib}: only {share:.1%} of counts land on in-vocab genes "
            f"(threshold {MIN_VOCAB_COVERAGE:.1%}) - check that {gene_col} holds the "
            f"right gene symbols for this library (common cause: wrong gene_col config "
            f"key, Ensembl IDs instead of symbols, or a species/annotation mismatch).",
            flush=True,
        )
    return share


def save_percell_gene_embeddings(adata, model, vocab, model_configs, gene_ids, gene_stats_dict_file, lib, outp):
    """For snRNA libraries: full per-cell per-gene embeddings, but only for a
    small bounded sample of cells (never the whole library). If gene_list is
    set, the output is restricted to just those genes (not "in addition to"
    the sampled cells - see README)."""
    n = adata.n_obs
    n_sample = min(N_SAMPLE_CELLS, n)
    rng = random.Random(SEED)
    idx = sorted(rng.sample(range(n), n_sample))
    sub = adata[idx].copy()
    region_labels = sub.obs_names.tolist()  # one "region" per sampled cell == per-cell output

    _, gene_adata, _ = gene_embed.get_batch_embeddings_with_genes(
        sub, gene_stats_dict_file, model, vocab, model_configs, gene_ids,
        region_labels, max_length=MAX_LENGTH, batch_size=BATCH_SIZE,
    )
    if GENE_LIST:
        gene_adata = gene_adata[gene_adata.obs["gene"].isin(GENE_LIST)].copy()
    gene_adata.write_h5ad(outp)
    return gene_adata


def run_library(lib):
    outp = f"{OUT}/{lib}_emb.h5ad"
    genes_outp = f"{OUT}/{lib}_genes.h5ad"
    percell_outp = f"{OUT}/{lib}_genes_percell.h5ad"

    adata = sc.read_h5ad(f"{DATA}/{lib}.h5ad")
    is_spatial, platform = detect_platform(adata, CFG)
    print(f"[{lib}] platform={platform} spatial={is_spatial}", flush=True)

    done = os.path.exists(outp) and os.path.exists(genes_outp)
    if is_spatial:
        if done:
            print(f"[skip] {lib} already done"); return
    else:
        if done and os.path.exists(percell_outp):
            print(f"[skip] {lib} already done"); return

    adata = apply_in_tissue_filter(adata, lib)
    n_before = adata.n_obs
    sc.pp.calculate_qc_metrics(adata, inplace=True)

    min_genes = resolve_min_genes(MIN_GENES_CFG, platform, adata.n_vars)
    adata = adata[adata.obs["n_genes_by_counts"] >= min_genes].copy()
    print(f"[{lib}] units kept {adata.n_obs}/{n_before} (>= {min_genes} genes, "
          f"min_genes={MIN_GENES_CFG})", flush=True)

    vocab_for_check = gene_embed.load_vocab(CKPT)
    adata, gene_col_used = select_gene_col_and_dedupe(adata, vocab_for_check, CFG, lib)
    print_vocab_coverage(lib, adata, vocab_for_check)

    obs_to_save = "library_id" if "library_id" in adata.obs else None
    t = time.time()
    out = embed_data(adata, CKPT, gene_col="feature_name", obs_to_save=obs_to_save,
                     batch_size=BATCH_SIZE, use_fast_transformer=False, return_new_adata=True,
                     device=DEVICE)
    secs_cell = time.time() - t
    assert out.n_obs == adata.n_obs, "unit count mismatch"
    if not (out.obs_names == adata.obs_names).all():
        print(f"[warn] {lib}: obs names differ; relying on row order")
    nan = int(np.isnan(out.X).sum())
    if "spatial" in adata.obsm:
        out.obsm["spatial"] = adata.obsm["spatial"]
    out.obs["n_genes"] = adata.obs["n_genes_by_counts"].values
    out.obs["log_total_counts"] = np.log1p(adata.obs["total_counts"].values)
    for c in OBS_COLS_TO_CARRY:
        if c in adata.obs:
            out.obs[c] = adata.obs[c].values

    sc.pp.neighbors(out, use_rep="X")
    sc.tl.leiden(out, resolution=LEIDEN_RES, key_added="leiden_coarse",
                 flavor="igraph", n_iterations=2, directed=False, random_state=SEED)

    if out.n_obs > MAX_UNITS_FOR_UMAP:
        print(f"[{lib}] {out.n_obs} units > max_units_for_umap ({MAX_UNITS_FOR_UMAP}); "
              f"UMAP/plots computed on a subsample, clustering kept for all units", flush=True)
        umap_rng = np.random.RandomState(SEED)
        sub_idx = umap_rng.choice(out.n_obs, size=MAX_UNITS_FOR_UMAP, replace=False)
        umap_src = out[sub_idx].copy()
        sc.tl.umap(umap_src, random_state=SEED)
        out.uns["umap_subsample_idx"] = sub_idx
        plot_target = umap_src
    else:
        sc.tl.umap(out, random_state=SEED)
        plot_target = out

    out.write_h5ad(outp)
    sc.pl.umap(plot_target, color="leiden_coarse", title=f"{lib} - scGPT-spatial zero-shot embedding",
               show=False, save=f"_{lib}.png")
    if is_spatial:
        sc.pl.embedding(out, basis="spatial", color=["leiden_coarse", "n_genes"],
                        show=False, save=f"_{lib}.png")

    # gene-level embeddings, aggregated per leiden cluster (region / cell-type proxy)
    t = time.time()
    model, vocab, model_configs, gene_stats_dict_file = gene_embed.load_model_and_vocab(CKPT, device=DEVICE)
    adata.var["id_in_vocab"] = [vocab[g] if g in vocab else -1 for g in adata.var["feature_name"]]
    adata_v = adata[:, adata.var["id_in_vocab"] >= 0].copy()
    gene_ids = np.array(vocab(adata_v.var["feature_name"].tolist()), dtype=int)
    region_labels = out.obs["leiden_coarse"].values

    _, gene_adata, _ = gene_embed.get_batch_embeddings_with_genes(
        adata_v, gene_stats_dict_file, model, vocab, model_configs, gene_ids,
        region_labels, max_length=MAX_LENGTH, batch_size=BATCH_SIZE,
        min_count=MIN_SPOTS_PER_GENE_REGION,
    )
    secs_gene = time.time() - t
    gene_nan = int(np.isnan(gene_adata.X).sum())
    gene_adata.write_h5ad(genes_outp)

    if not is_spatial:
        save_percell_gene_embeddings(
            adata_v, model, vocab, model_configs, gene_ids, gene_stats_dict_file, lib, percell_outp
        )

    new = not os.path.exists(LOG)
    with open(LOG, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["library", "platform", "units_before", "units_kept", "cell_seconds", "gene_seconds", "nans", "gene_nans"])
        w.writerow([lib, platform, n_before, out.n_obs, round(secs_cell), round(secs_gene), nan, gene_nan])
    print(f"[done] {lib}: {out.shape} cells {secs_cell/60:.1f}min, genes {gene_adata.shape} {secs_gene/60:.1f}min, "
          f"NaNs {nan}/{gene_nan}", flush=True)


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    for lib in sys.argv[1:]:
        run_library(lib)


if __name__ == "__main__":
    main()
