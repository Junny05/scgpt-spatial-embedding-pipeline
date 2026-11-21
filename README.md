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
- `{lib}_genes.h5ad` — one row per `(region, gene)` pair seen in the data.
  `X` is the averaged 512-d contextualized gene embedding (post-transformer,
  self-attention over all genes in that spot/cell) for that gene within that
  region, aggregated via a running sum/count — a full per-spot gene matrix is
  never held in memory. `obs["region"]`/`obs["gene"]` identify each row; `NaN`
  rows mean that gene was never observed in that region, not a bug.
- `{lib}_genes_percell.h5ad` (snRNA libraries only) — the same per-gene
  embedding but one row per individual cell instead of per cluster, for a
  bounded sample of `n_sample_cells` cells (default 50) plus any genes in
  `gene_list`, never the whole library (full per-cell storage would be ~2.4MB
  per cell).
- `batch_log.csv` — per-library timing and NaN counts, appended across runs.
- `figures/*.png` — spatial scatter plots (spatial libraries only).

## Verification performed

- `[check] model params: N | missing from checkpoint: 0` confirms full
  checkpoint loading for both the existing cell-embedding path and the new
  gene-embedding path (`gene_embed.load_model_and_vocab`, which reuses
  `cell_emb.load_pretrained`).
- Gene-level output has 0 NaNs among genes actually observed in a region;
  shapes are `(n_regions * n_genes_seen, 512)`.
- Gene-id → gene-name mapping was independently verified: for a region/gene
  pair, the vocab id looked up via `GeneVocab` and the running sum/count
  buffer in `GeneAggregator` reproduce exactly the row written to
  `{lib}_genes.h5ad` (see test in plan/verification notes — not re-run on
  every invocation).

## Known limitations

- "Region" is a leiden cluster computed from the cell/spot embeddings within
  a single library, not a true anatomical annotation — it's a proxy grouping,
  and resolution (`leiden_resolution` in config) changes cluster granularity.
- Genes not present in the scGPT vocabulary are silently dropped (logged by
  `embed_data`/`id_in_vocab` filtering); gene symbol mismatches (e.g. wrong
  `feature_name` column) will silently reduce coverage rather than error.
- Per-cell snRNA gene embeddings are only saved for a small sample of cells,
  not the full library, to keep output size bounded.
- Zero-shot only; no fine-tuning or batch-effect correction across libraries.
- Verified on CPU (Apple M3); GPU path is untouched (`device` defaults to
  `cuda` when available) but not re-verified here.
