"""
Fork of scgpt_spatial.tasks.cell_emb.get_batch_cell_embeddings that also
captures the per-gene contextualized embedding (the full model._encode()
output, not just the <cls> row) and aggregates it on the fly into a running
sum/count per region, instead of ever materializing a (genes x 512) matrix
per spot/cell for a whole library.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm

from scgpt_spatial.data_collator import DataCollator
from scgpt_spatial.model import TransformerModel
from scgpt_spatial.tokenizer import GeneVocab
from scgpt_spatial.tasks.cell_emb import load_pretrained


class GeneAggregator:
    """Running sum/count of per-gene embeddings, keyed by region label.

    Never stores a per-spot gene matrix; each batch is scatter-added into
    the (vocab_size, embsize) running buffers for whatever regions it
    touches, then discarded.
    """

    def __init__(self, vocab_size: int, embsize: int):
        self.vocab_size = vocab_size
        self.embsize = embsize
        self.sums: dict[str, np.ndarray] = {}
        self.counts: dict[str, np.ndarray] = {}

    def _buffers(self, region: str):
        if region not in self.sums:
            self.sums[region] = np.zeros((self.vocab_size, self.embsize), dtype=np.float64)
            self.counts[region] = np.zeros(self.vocab_size, dtype=np.int64)
        return self.sums[region], self.counts[region]

    def add(self, region_labels, gene_ids: np.ndarray, gene_embs: np.ndarray, mask: np.ndarray):
        """
        region_labels: (batch,) region label per spot/cell in this batch
        gene_ids: (batch, seq_len-1) vocab ids, no <cls>
        gene_embs: (batch, seq_len-1, embsize)
        mask: (batch, seq_len-1) bool, True where real gene token (not padding)
        """
        for i, region in enumerate(region_labels):
            sum_buf, count_buf = self._buffers(str(region))
            valid = mask[i]
            ids = gene_ids[i, valid]
            embs = gene_embs[i, valid].astype(np.float64)
            np.add.at(sum_buf, ids, embs)
            np.add.at(count_buf, ids, 1)

    def finalize(self, vocab: GeneVocab, min_count: int = 1):
        """Return an AnnData with one row per (region, gene) pair that was
        actually observed at least `min_count` times in that region, X = the
        averaged 512-d embedding. Rows below `min_count` (including genes
        never seen in a region) are dropped rather than kept as NaN, so the
        output only contains means that are actually comparable and the file
        doesn't balloon with empty rows. obs["n_obs"] carries the number of
        spots/cells each row was averaged over, so a mean from 2 spots isn't
        silently treated the same as one from 800.
        """
        itos = vocab.get_itos()
        regions = list(self.sums.keys())

        rows_X, rows_region, rows_gene, rows_n = [], [], [], []
        for region in regions:
            count_buf = self.counts[region]
            sum_buf = self.sums[region]
            keep_idx = np.where(count_buf >= min_count)[0]
            if len(keep_idx) == 0:
                continue
            avg = (sum_buf[keep_idx] / count_buf[keep_idx][:, None]).astype(np.float32)
            rows_X.append(avg)
            rows_region.extend([region] * len(keep_idx))
            rows_gene.extend(itos[i] for i in keep_idx)
            rows_n.extend(count_buf[keep_idx].tolist())

        import anndata as ad

        X = np.concatenate(rows_X, axis=0) if rows_X else np.zeros((0, self.embsize), dtype=np.float32)
        obs = pd.DataFrame({"region": rows_region, "gene": rows_gene, "n_obs": rows_n})
        return ad.AnnData(X=X, obs=obs)


def get_batch_embeddings_with_genes(
    adata,
    gene_stats_dict_file,
    model,
    vocab,
    model_configs,
    gene_ids,
    region_labels,
    max_length=1200,
    batch_size=16,
    min_count=1,
):
    """
    Same forward pass as scgpt_spatial.tasks.cell_emb.get_batch_cell_embeddings
    (cell_embedding_mode="cls", use_batch_labels=False), but also returns a
    finalized gene-level AnnData aggregated per region via GeneAggregator.

    region_labels must be aligned row-for-row with adata (same order as
    count_matrix / gene_ids).
    """
    count_matrix = adata.X
    count_matrix = count_matrix if isinstance(count_matrix, np.ndarray) else count_matrix.A
    region_labels = np.asarray(region_labels)
    assert len(region_labels) == count_matrix.shape[0]

    class Dataset(torch.utils.data.Dataset):
        def __init__(self, count_matrix, gene_ids, gene_stats_dict_file):
            self.slide_mean = np.mean(
                count_matrix[count_matrix.nonzero()[0], count_matrix.nonzero()[1]]
            )
            self.count_matrix = count_matrix / self.slide_mean
            self.gene_ids = gene_ids
            self.gene_stats_dict = pd.read_csv(gene_stats_dict_file, index_col=0)
            new_genes = set(self.gene_ids).difference(set(self.gene_stats_dict.index.values))
            for i in new_genes:
                idx = np.where(self.gene_ids == i)[0]
                col = self.count_matrix[:, idx].flatten()
                nonzero_idx = np.nonzero(col)[0]
                values = col[nonzero_idx]
                self.gene_stats_dict.loc[i] = [float(values.mean())]

        def __len__(self):
            return len(self.count_matrix)

        def __getitem__(self, idx):
            row = self.count_matrix[idx]
            nonzero_idx = np.nonzero(row)[0]
            values = row[nonzero_idx]
            genes = self.gene_ids[nonzero_idx]
            mean_divide_by = self.gene_stats_dict.loc[genes, "mean"].values
            values = np.divide(values, mean_divide_by)
            genes = np.insert(genes, 0, vocab["<cls>"])
            values = np.insert(values, 0, model_configs["pad_value"])
            return {
                "id": idx,
                "genes": torch.from_numpy(genes).long(),
                "expressions": torch.from_numpy(values).float(),
            }

    dataset = Dataset(count_matrix, gene_ids, gene_stats_dict_file)
    collator = DataCollator(
        do_padding=True,
        pad_token_id=vocab[model_configs["pad_token"]],
        pad_value=model_configs["pad_value"],
        do_mlm=False,
        do_binning=True,
        n_bins=51,
        max_length=max_length,
        sampling=True,
        keep_first_n_tokens=1,
    )
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=SequentialSampler(dataset),
        collate_fn=collator,
        drop_last=False,
        num_workers=0,
        pin_memory=True,
    )

    device = next(model.parameters()).device
    embsize = model_configs["embsize"]
    cell_embeddings = np.zeros((len(dataset), embsize), dtype=np.float32)
    aggregator = GeneAggregator(vocab_size=len(vocab), embsize=embsize)

    with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
        count = 0
        for data_dict in tqdm(data_loader, desc="Embedding cells + genes"):
            input_gene_ids = data_dict["gene"].to(device)
            src_key_padding_mask = input_gene_ids.eq(vocab[model_configs["pad_token"]])

            output = model._encode(
                input_gene_ids,
                data_dict["expr"].to(device),
                src_key_padding_mask=src_key_padding_mask,
                batch_labels=None,
            )

            cls_emb = output[:, 0, :].cpu().numpy()
            cell_embeddings[count : count + len(cls_emb)] = cls_emb

            gene_embs = output[:, 1:, :].cpu().numpy()
            gene_ids_batch = input_gene_ids[:, 1:].cpu().numpy()
            pad_mask = src_key_padding_mask[:, 1:].cpu().numpy()
            valid_mask = ~pad_mask

            batch_regions = region_labels[count : count + len(cls_emb)]
            aggregator.add(batch_regions, gene_ids_batch, gene_embs, valid_mask)

            count += len(cls_emb)

    cell_embeddings = cell_embeddings / np.linalg.norm(cell_embeddings, axis=1, keepdims=True)
    gene_adata = aggregator.finalize(vocab, min_count=min_count)
    return cell_embeddings, gene_adata, aggregator


def load_vocab(model_dir):
    """Lightweight: vocab only, no model weights. Used for pre-run diagnostics."""
    model_dir = Path(model_dir)
    vocab = GeneVocab.from_file(model_dir / "vocab.json")
    for s in ["<pad>", "<cls>", "<eoc>"]:
        if s not in vocab:
            vocab.append_token(s)
    vocab.set_default_index(vocab["<pad>"])
    return vocab


def load_model_and_vocab(model_dir, device="cpu", use_fast_transformer=False):
    model_dir = Path(model_dir)
    vocab_file = model_dir / "vocab.json"
    model_config_file = model_dir / "args.json"
    model_file = model_dir / "best_model.pt"
    gene_stats_dict_file = model_dir / "all_dict_mean_std.csv"
    pad_token = "<pad>"
    special_tokens = [pad_token, "<cls>", "<eoc>"]

    vocab = GeneVocab.from_file(vocab_file)
    for s in special_tokens:
        if s not in vocab:
            vocab.append_token(s)
    vocab.set_default_index(vocab["<pad>"])

    with open(model_config_file) as f:
        model_configs = json.load(f)

    model = TransformerModel(
        ntoken=len(vocab),
        d_model=model_configs["embsize"],
        nhead=model_configs["nheads"],
        d_hid=model_configs["d_hid"],
        nlayers=model_configs["nlayers"],
        nlayers_cls=model_configs["n_layers_cls"],
        n_cls=1,
        vocab=vocab,
        dropout=model_configs["dropout"],
        pad_token=model_configs["pad_token"],
        pad_value=model_configs["pad_value"],
        do_mvc=True,
        do_dab=False,
        use_generative_training=False,
        use_batch_labels=False,
        explicit_zero_prob=False,
        use_fast_transformer=use_fast_transformer,
        fast_transformer_backend="flash",
        pre_norm=False,
        use_MVC_impute=True,
        use_moe_dec=True,
    )
    load_pretrained(model, torch.load(model_file, map_location=device), verbose=False)
    model.to(device)
    model.eval()
    return model, vocab, model_configs, gene_stats_dict_file
