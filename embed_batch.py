import sys, os, time, csv, random
import numpy as np
import scanpy as sc

from config import load_config

CFG = load_config(os.environ.get("SCGPT_CONFIG", "config.yaml"))
REPO = CFG["repo"]
CKPT = CFG["checkpoint"]
DATA = CFG["data_dir"]
OUT = CFG["out_dir"]
LOG = CFG["log_file"]
MIN_GENES = CFG["min_genes"]
BATCH_SIZE = CFG["batch_size"]
MAX_LENGTH = CFG["max_length"]
LEIDEN_RES = CFG["leiden_resolution"]
N_SAMPLE_CELLS = CFG["n_sample_cells"]
GENE_LIST = set(CFG["gene_list"])

sys.path.insert(0, REPO)
from scgpt_spatial.tasks import embed_data  # noqa: E402

import gene_embed  # noqa: E402

OBS_COLS_TO_CARRY = [
    "clinical_group", "tissue_source", "subject_id", "region", "sample_id",
    "Batch", "Source", "individualID", "libraryID",
]


def prep_gene_col_snrna(adata):
    """snRNA: use var['gene_name'] as the gene symbol column. Drop genes with
    no name, and for duplicate symbols keep the one with the highest total
    counts (computed on the cells currently in adata)."""
    gene_name = adata.var["gene_name"]
    valid = gene_name.notna() & (gene_name.astype(str).str.len() > 0)
    adata = adata[:, valid.values].copy()

    counts_per_gene = np.asarray(adata.X.sum(axis=0)).flatten()
    adata.var["_total_counts_now"] = counts_per_gene
    adata.var["feature_name"] = adata.var["gene_name"].values

    order = adata.var.sort_values("_total_counts_now", ascending=False).index
    dup = adata.var.loc[order, "feature_name"].duplicated(keep="first")
    keep = order[~dup]
    # restore original var order among kept genes
    keep_sorted = [v for v in adata.var_names if v in set(keep)]
    adata = adata[:, keep_sorted].copy()
    adata.var.drop(columns=["_total_counts_now"], inplace=True)
    return adata


def print_vocab_coverage(lib, adata, vocab, gene_col="feature_name"):
    genes = adata.var[gene_col].astype(str)
    in_vocab = genes.isin(vocab.get_itos())
    counts_per_gene = np.asarray(adata.X.sum(axis=0)).flatten()
    total_counts = counts_per_gene.sum()
    in_vocab_counts = counts_per_gene[in_vocab.values].sum()
    expressed_in_vocab = int(((counts_per_gene > 0) & in_vocab.values).sum())

    print(
        f"[{lib}] vocab coverage: genes_in_vocab={int(in_vocab.sum())}/{len(genes)} | "
        f"expressed_genes_in_vocab={expressed_in_vocab} | "
        f"share_of_counts_on_in_vocab_genes="
        f"{(in_vocab_counts / total_counts if total_counts > 0 else float('nan')):.4f}",
        flush=True,
    )


def save_percell_gene_embeddings(adata, model, vocab, model_configs, gene_ids, gene_stats_dict_file, lib, outp):
    """For snRNA libraries: full per-cell per-gene embeddings, but only for a
    small bounded sample of cells (never the whole library)."""
    n = adata.n_obs
    n_sample = min(N_SAMPLE_CELLS, n)
    rng = random.Random(0)
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


for lib in sys.argv[1:]:
    outp = f"{OUT}/{lib}_emb.h5ad"
    genes_outp = f"{OUT}/{lib}_genes.h5ad"
    percell_outp = f"{OUT}/{lib}_genes_percell.h5ad"

    adata = sc.read_h5ad(f"{DATA}/{lib}.h5ad")
    is_spatial = "spatial" in adata.obsm and "in_tissue" in adata.obs

    done = os.path.exists(outp) and os.path.exists(genes_outp)
    if is_spatial:
        if done:
            print(f"[skip] {lib} already done"); continue
    else:
        if done and os.path.exists(percell_outp):
            print(f"[skip] {lib} already done"); continue

    if "in_tissue" in adata.obs:
        adata = adata[adata.obs["in_tissue"] == 1].copy()
    n_before = adata.n_obs
    sc.pp.calculate_qc_metrics(adata, inplace=True)
    adata = adata[adata.obs["n_genes_by_counts"] >= MIN_GENES].copy()
    print(f"[{lib}] units kept {adata.n_obs}/{n_before} (>= {MIN_GENES} genes)", flush=True)

    if is_spatial:
        adata.var["feature_name"] = adata.var_names
    else:
        adata = prep_gene_col_snrna(adata)

    vocab_for_check = gene_embed.load_vocab(CKPT)
    print_vocab_coverage(lib, adata, vocab_for_check)

    obs_to_save = "library_id" if "library_id" in adata.obs else None
    t = time.time()
    out = embed_data(adata, CKPT, gene_col="feature_name", obs_to_save=obs_to_save,
                     batch_size=BATCH_SIZE, use_fast_transformer=False, return_new_adata=True)
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
                 flavor="igraph", n_iterations=2, directed=False)
    sc.tl.umap(out)
    out.write_h5ad(outp)
    sc.pl.umap(out, color="leiden_coarse", title=f"{lib} - scGPT-spatial zero-shot embedding",
               show=False, save=f"_{lib}.png")
    if is_spatial:
        sc.pl.embedding(out, basis="spatial", color=["leiden_coarse", "n_genes"],
                        show=False, save=f"_{lib}.png")

    # gene-level embeddings, aggregated per leiden cluster (region / cell-type proxy)
    t = time.time()
    model, vocab, model_configs, gene_stats_dict_file = gene_embed.load_model_and_vocab(CKPT)
    adata.var["id_in_vocab"] = [vocab[g] if g in vocab else -1 for g in adata.var["feature_name"]]
    adata_v = adata[:, adata.var["id_in_vocab"] >= 0].copy()
    gene_ids = np.array(vocab(adata_v.var["feature_name"].tolist()), dtype=int)
    region_labels = out.obs["leiden_coarse"].values

    _, gene_adata, _ = gene_embed.get_batch_embeddings_with_genes(
        adata_v, gene_stats_dict_file, model, vocab, model_configs, gene_ids,
        region_labels, max_length=MAX_LENGTH, batch_size=BATCH_SIZE,
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
            w.writerow(["library", "units_before", "units_kept", "cell_seconds", "gene_seconds", "nans", "gene_nans"])
        w.writerow([lib, n_before, out.n_obs, round(secs_cell), round(secs_gene), nan, gene_nan])
    print(f"[done] {lib}: {out.shape} cells {secs_cell/60:.1f}min, genes {gene_adata.shape} {secs_gene/60:.1f}min, "
          f"NaNs {nan}/{gene_nan}", flush=True)
