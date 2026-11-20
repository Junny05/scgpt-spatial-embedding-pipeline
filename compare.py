import glob, os, numpy as np, scanpy as sc, anndata as ad
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

from config import load_config

CFG = load_config(os.environ.get("SCGPT_CONFIG", "config.yaml"))
files = sorted(glob.glob(f"{CFG['out_dir']}/*_emb.h5ad"))
parts = []
for f in files:
    a = sc.read_h5ad(f)
    lib = os.path.basename(f).replace("_emb.h5ad", "")
    a.obs["library"] = lib
    a.obs["group"] = a.obs["clinical_group"].astype(str).values if "clinical_group" in a.obs else "NA"
    a.obs["subject"] = a.obs["subject_id"].astype(str).values if "subject_id" in a.obs else "NA"
    a.obsm.clear(); a.obsp.clear(); a.uns.clear()
    a.obs_names = [f"{lib}_{i}" for i in range(a.n_obs)]
    parts.append(a[:, :])
all_ = ad.concat(parts, join="inner")
print(all_.shape, all_.obs.groupby(["library", "group", "subject"]).size())

sc.pp.neighbors(all_, use_rep="X", transformer="sklearn")
sc.tl.umap(all_)
sc.pl.umap(all_, color=["library", "group"], show=False, save="_all_libraries.png")

X = all_.X
rng = np.random.RandomState(0)
idx = rng.choice(all_.n_obs, size=min(6000, all_.n_obs), replace=False)
for lab in ["library", "group"]:
    s = silhouette_score(X[idx], all_.obs[lab].values[idx])
    print(f"silhouette by {lab}: {s:.3f}")

nn = NearestNeighbors(n_neighbors=16).fit(X)
_, nbr = nn.kneighbors(X[idx])
labs = all_.obs["library"].values
same = np.mean([(labs[nbr[i, 1:]] == labs[idx[i]]).mean() for i in range(len(idx))])
print(f"mean share of 15 nearest neighbors from the same library: {same:.2f} (0.25 = fully mixed)")
