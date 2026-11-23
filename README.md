# scGPT-spatial embedding pipeline

Runs scGPT-spatial (zero-shot) on spatial and single-nucleus RNA-seq `.h5ad`
libraries, producing both spot/cell-level embeddings and gene-level
embeddings.

## Platform support

| Platform | Status | Notes |
| --- | --- | --- |
| 10x Visium | tested | original target platform; verified end-to-end on real data |
| snRNA-seq | tested | verified end-to-end on real data |
| 10x Visium HD | untested | code path implemented (no `in_tissue` filter required, sparse-matrix path for large unit counts); not run against real Visium HD data |
| Xenium | untested | targeted-panel code path implemented (auto `min_genes`, gene_col auto-selection); not run against real Xenium data |
| MERFISH | untested | same as Xenium; not run against real MERFISH data |

Platform is auto-detected per library (`obsm["spatial"]` presence, panel
size, `obs["in_tissue"]` presence) and logged for every run; override with
the `platform` config key if auto-detection picks the wrong one.

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

Spatial vs. snRNA is auto-detected from `obsm["spatial"]`; the `in_tissue`
filter is applied only if `obs["in_tissue"]` exists (Visium HD libraries
commonly have no `in_tissue` column and are still spatial). The gene-symbol
column is auto-selected per library by comparing `var["gene_name"]`,
`var["gene_symbols"]`, `var["feature_name"]`, and `var_names` against the
model vocabulary and picking the best-overlapping one (overridable via
`gene_col`); genes are deduped by symbol (keeping the highest-total-count
row) for every platform. `min_genes` defaults to `"auto"`: 200 for
whole-transcriptome data, a 5%-of-panel-size floor (minimum 5) for targeted
panels like Xenium/MERFISH.

Any ambiguity (e.g. platform can't be inferred, gene_col overlap too low)
raises an error naming the config key that resolves it, rather than a bare
`KeyError`.

## Outputs (in `out_dir`)

- `{lib}_emb.h5ad` — one 512-d embedding per spot/cell (`X`), plus QC stats,
  metadata, a `leiden_coarse` cluster label used as the "region" for
  gene-level aggregation, and UMAP coordinates in `obsm["X_umap"]`. For
  libraries above `max_units_for_umap` (default 50,000 — relevant to large
  Visium HD/Xenium/MERFISH libraries), clustering (neighbors + leiden) still
  runs on every unit, but UMAP and the UMAP plot are computed only on a
  fixed-seed subsample of that size; `out.write_h5ad` is then **not** holding
  a full `X_umap` for every unit in that case (`obsm["X_umap_subsample_idx"]`
  records which rows were used for the UMAP plot).
- `figures/umap_{lib}.png` — UMAP colored by `leiden_coarse` (full library,
  or the UMAP subsample above the size cutoff).
- `{lib}_genes.h5ad` — one row per `(region, gene)` pair that was observed at
  least `min_spots_per_gene_region` times (default 3) in that region. `X` is
  the averaged 512-d contextualized gene embedding (post-transformer,
  self-attention over all genes in that spot/cell) for that gene within that
  region, aggregated via a running sum/count (float32) — a full per-spot
  gene matrix is never held in memory, and the count matrix itself is read
  as a sparse matrix on demand rather than densified. `obs["region"]`/
  `obs["gene"]` identify each row, and `obs["n_obs"]` is how many
  spots/cells the row was averaged over, so a mean from 3 spots isn't
  silently treated as comparable to one from 800. Rows below the cutoff
  (including genes never seen in a region) are dropped rather than written
  as NaN, to keep file size down.
- `{lib}_genes_percell.h5ad` (non-spatial/snRNA libraries only) — the same
  per-gene embedding but one row per individual cell instead of per cluster,
  for a bounded sample of `n_sample_cells` cells (default 50). If `gene_list`
  is set in config, the output is **restricted** to only those genes (it does
  not add genes on top of the `n_sample_cells` sample — this is a filter, not
  an addition).
- `batch_log.csv` — per-library platform, timing, and NaN counts, appended
  across runs. The model runs twice per library (once for cell embeddings,
  once for gene embeddings), so wall-clock runtime for a run is roughly
  `cell_seconds + gene_seconds` from this file, not either figure alone.
- `figures/*.png` — spatial scatter plots (spatial libraries only).

## Reproducibility

`seed` in config (default 0) seeds Python's `random`, NumPy, and PyTorch
before each run, and is used directly for leiden/UMAP `random_state` and for
the UMAP/per-cell-sample subsampling. It also seeds the global PyTorch RNG
used internally by `scgpt_spatial`'s `DataCollator` when a cell/spot has more
genes than `max_length` and must be randomly subsampled — that subsampling
code is in the vendored `scGPT-spatial` package and isn't itself
configurable, so this is the only way to make those runs reproducible run to
run with the same inputs.

## Known limitations

- "Region" is a leiden cluster computed from the cell/spot embeddings within
  a single library, not a true anatomical annotation — it's a proxy grouping,
  and resolution (`leiden_resolution` in config) changes cluster granularity.
- Genes not present in the scGPT vocabulary are dropped. This is not silent:
  every run prints `genes_in_vocab`, `expressed_genes_in_vocab`, and
  `share_of_counts_on_in_vocab_genes` for that library, and prints a
  `[WARNING]` if the share falls below `min_vocab_coverage` (default 0.90) —
  the usual cause is a wrong `gene_col` (e.g. Ensembl IDs instead of
  symbols, or a species/annotation mismatch).
- `tests/test_checkpoint_loading.py` loads the real checkpoint and asserts
  `missing from checkpoint: 0` — run it with `pytest tests/` after setting
  up `config.yaml` (it's skipped, not failed, if `config.yaml` doesn't exist
  yet). The other tests in `tests/` use small synthetic AnnData and run
  without a checkpoint or `config.yaml`.
- Per-cell snRNA gene embeddings are only saved for a small sample of cells,
  not the full library, to keep output size bounded.
- Zero-shot only; no fine-tuning or batch-effect correction across libraries.
- Verified end-to-end on real data for Visium and snRNA only, on CPU (Apple
  M3). Visium HD, Xenium, and MERFISH support is implemented (platform
  detection, targeted-panel QC defaults, sparse-matrix handling for large
  unit counts) but **untested** against real data from those platforms —
  see the platform table above. GPU path (`device: cuda`) is implemented but
  not independently re-verified here.
