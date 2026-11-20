import os
import yaml

REQUIRED_KEYS = ["repo", "checkpoint", "data_dir", "out_dir", "log_file"]


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)

    missing = [k for k in REQUIRED_KEYS if k not in cfg]
    if missing:
        raise ValueError(f"config missing required keys: {missing}")

    for key in ["repo", "checkpoint", "data_dir"]:
        if not os.path.exists(cfg[key]):
            raise FileNotFoundError(f"config[{key}] does not exist: {cfg[key]}")

    os.makedirs(cfg["out_dir"], exist_ok=True)
    cfg.setdefault("min_genes", 200)
    cfg.setdefault("batch_size", 16)
    cfg.setdefault("max_length", 1200)
    cfg.setdefault("leiden_resolution", 0.15)
    cfg.setdefault("n_sample_cells", 50)
    cfg.setdefault("gene_list", [])
    return cfg
