# scGPT-spatial embedding pipeline

Runs scGPT-spatial (zero-shot) on Visium spatial and single-nucleus RNA-seq
`.h5ad` libraries, producing both spot/cell-level embeddings and gene-level
embeddings.

## Setup

0. Python 3.9, `pip install -r requirements.txt`.
1. This repo includes a CPU-patched copy of scGPT-spatial under
   `scGPT-spatial/` (optional `flash_attn`, `use_fast_transformer=False`,
   `Wqkv.` → `in_proj_` checkpoint key remap, optional `scib`,
   `num_workers=0` — lets it run without a GPU; see
   `scGPT-spatial/README.md` for the upstream project).
2. Download the pretrained checkpoint (`vocab.json`, `args.json`,
   `best_model.pt`, `all_dict_mean_std.csv`) from the source linked in
   `scGPT-spatial/README.md` (not committed — it's ~220MB, over GitHub's
   file limit) and place it under
   `scGPT-spatial/checkpoints/scGPT_spatial_v1/`.
3. ```
   cp config.example.yaml config.yaml
   # edit config.yaml: repo (path to scGPT-spatial/ above), checkpoint, data_dir, out_dir
   ```

`config.yaml` is gitignored — never commit it or any `.h5ad` file.

## Run

```
python embed_batch.py <library_name> [<library_name> ...]
```

Each `<library_name>` must have a matching `<data_dir>/<library_name>.h5ad`.
A library already fully processed is skipped automatically (resumable).

Spatial vs. snRNA is auto-detected by the presence of `obsm["spatial"]` and
`obs["in_tissue"]`.

## Outputs (in `out_dir`)

- `{lib}_emb.h5ad` — one 512-d embedding per spot/cell (`X`), plus QC stats,
  metadata, a `leiden_coarse` cluster label used as the "region" for
  gene-level aggregation, and UMAP coordinates in `obsm["X_umap"]`.
- `figures/umap_{lib}.png` — UMAP colored by `leiden_coarse`, generated every
  run from the same `out` AnnData written to `{lib}_emb.h5ad`.
- `{lib}_genes.h5ad` — one row per `(region, gene)` pair that was observed at
  least `min_spots_per_gene_region` times (default 3) in that region. `X` is
  the averaged 512-d contextualized gene embedding (post-transformer,
  self-attention over all genes in that spot/cell) for that gene within that
  region, aggregated via a running sum/count — a full per-spot gene matrix is
  never held in memory. `obs["region"]`/`obs["gene"]` identify each row,
  and `obs["n_obs"]` is how many spots/cells the row was averaged over, so a
  mean from 3 spots isn't silently treated as comparable to one from 800.
  Rows below the cutoff (including genes never seen in a region) are dropped
  rather than written as NaN, to keep file size down.
- `{lib}_genes_percell.h5ad` (snRNA libraries only) — the same per-gene
  embedding but one row per individual cell instead of per cluster, for a
  bounded sample of `n_sample_cells` cells (default 50) plus any genes in
  `gene_list`, never the whole library (full per-cell storage would be ~2.4MB
  per cell).
- `batch_log.csv` — per-library timing and NaN counts, appended across runs.
- `figures/*.png` — spatial scatter plots (spatial libraries only).

## Verification performed

- `tests/test_checkpoint_loading.py` loads the real checkpoint and asserts
  `missing from checkpoint: 0` — run it with `pytest tests/` after setting
  up `config.yaml` (it's skipped, not failed, if `config.yaml` doesn't exist
  yet). This catches a silent weight-loading failure if the model or
  checkpoint ever changes incompatibly.
- Gene-level output has 0 NaNs (rows below `min_spots_per_gene_region` are
  dropped rather than kept as NaN).
- Gene-id → gene-name mapping was independently verified: for a region/gene
  pair, the vocab id looked up via `GeneVocab` and the running sum/count
  buffer in `GeneAggregator` reproduce exactly the row written to
  `{lib}_genes.h5ad`.

## Known limitations

- "Region" is a leiden cluster computed from the cell/spot embeddings within
  a single library, not a true anatomical annotation — it's a proxy grouping,
  and resolution (`leiden_resolution` in config) changes cluster granularity.
- Genes not present in the scGPT vocabulary are dropped (`embed_data`'s
  `id_in_vocab` filtering). This is no longer silent: every run prints
  `genes_in_vocab`, `expressed_genes_in_vocab`, and
  `share_of_counts_on_in_vocab_genes` for that library, and prints a
  `[WARNING]` if the share falls below `min_vocab_coverage` (default 0.90) —
  the usual cause is a wrong `gene_col`/var column (e.g. Ensembl IDs instead
  of symbols, or a species/annotation mismatch).
- Per-cell snRNA gene embeddings are only saved for a small sample of cells,
  not the full library, to keep output size bounded.
- Zero-shot only; no fine-tuning or batch-effect correction across libraries.
- Verified on CPU (Apple M3); GPU path is untouched (`device` defaults to
  `cuda` when available) but not re-verified here.
