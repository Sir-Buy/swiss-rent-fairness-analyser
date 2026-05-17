"""
cluster.py — hierarchical clustering of clean apartments into k=6 market tiers.

ML rationale (the parts that will be questioned):

  Features (chf_per_m2, rooms, latitude, longitude), all StandardScaler-d so
  no single feature dominates the Euclidean distance (raw lat/lon ≈ 46-48
  would otherwise drown chf_per_m2 ≈ 20-30 and rooms ≈ 1-6).

  Linkage = Ward. Ward minimises total within-cluster variance, producing
  compact, similarly-sized clusters that are well-behaved with Euclidean
  distance on standardised features. Average linkage was considered and
  rejected because Ward's similarly-sized clusters give each per-cluster
  regression in Step 4 a usable sample size.

  Cut at k=6 (fcluster maxclust). Six is chosen as a balance between
  interpretability (six colours stay distinguishable, six rows fit on a
  legend) and geographic granularity (Switzerland's rental market has
  roughly six tiers: Geneva-tier urban, Zurich-tier urban, mid-tier urban,
  suburban, regional, rural).

  Why hierarchical over k-means: deterministic (no random init), the
  dendrogram is a defensible artifact, and we don't have to assume
  spherical clusters in lat/lon space (the Rhone valley is not spherical).

  Hybridity caveat: clusters are a mix of geographic and price-tier
  signal. Two listings can land in the same cluster either because they're
  nearby OR because they have similar price/size. Intentional, disclosed
  in the Method tab.

Outputs:
  data/clustered.csv      clean.csv + 'cluster' column (1-6)
  data/dendrogram.png     truncated dendrogram (last 30 merges), shared palette
  data/cluster_summary.csv  cluster, count, mean_chf_per_m2, mean_rooms, dominant_canton
  data/model.pkl          fitted StandardScaler + feature list (Step 4 extends)

Reproducibility: rows are sorted by id before linkage so the dendrogram
is identical across runs even if raw row order changes.

Diagnostic: silhouette_score(k=6) is printed for the record — NOT used to
justify k=6 (the spec mandates it). It's there so the question "how did
you check k=6 was reasonable?" has a number in hand.

Usage:
    python cluster.py
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless: render the PNG without a display backend
import matplotlib.pyplot as plt
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage, set_link_color_palette
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


DATA_DIR = Path(__file__).parent / "data"
CLEAN_CSV = DATA_DIR / "clean.csv"
CLUSTERED_CSV = DATA_DIR / "clustered.csv"
DENDROGRAM_PNG = DATA_DIR / "dendrogram.png"
CLUSTER_SUMMARY_CSV = DATA_DIR / "cluster_summary.csv"
CLUSTER_DIAGNOSTICS_CSV = DATA_DIR / "cluster_diagnostics.csv"
MODEL_PKL = DATA_DIR / "model.pkl"

# Range of k values for the silhouette diagnostic. We report neighbouring
# values so anyone reviewing the project can see whether k=6 is the local
# best or whether a different k would have been a stronger pick.
K_DIAGNOSTIC_RANGE = (4, 5, 6, 7, 8)

FEATURES = ["chf_per_m2", "rooms", "latitude", "longitude"]
K = 6

# Shared palette: 6 evenly-spaced viridis samples. Re-imported by app.py for
# the Tab 1 map markers so cluster N always shows the same colour across the
# dendrogram, summary table, and folium markers.
PALETTE = ["#440154", "#414487", "#2a788e", "#22a884", "#7ad151", "#fde725"]


def main() -> None:
    if not CLEAN_CSV.exists():
        raise SystemExit(f"missing input: {CLEAN_CSV}. Run clean.py first.")

    df = pd.read_csv(CLEAN_CSV)
    df = df.sort_values("id").reset_index(drop=True)
    print(f"Loaded {len(df)} rows from {CLEAN_CSV.name}")

    X_raw = df[FEATURES].to_numpy(dtype=float)
    scaler = StandardScaler().fit(X_raw)
    X = scaler.transform(X_raw)

    print(f"Linkage (method='ward', n={len(X)}) ...")
    Z = linkage(X, method="ward")

    labels = fcluster(Z, t=K, criterion="maxclust")
    df["cluster"] = labels
    df.to_csv(CLUSTERED_CSV, index=False)
    print(f"Saved cluster assignments -> {CLUSTERED_CSV.name}")

    sil_k6 = silhouette_score(X, labels, metric="euclidean")
    print(f"\nSilhouette score at k={K}: {sil_k6:.4f}")
    print("  (range -1..+1; positive = well-separated, ~0 = overlapping)")

    # k-sweep diagnostic — compute silhouette at neighbouring k values so the
    # k=6 choice is auditable. The linkage matrix Z is reused; only fcluster
    # and silhouette are recomputed per k.
    print(f"\nk-sweep silhouette diagnostic over {K_DIAGNOSTIC_RANGE}:")
    diag_rows = []
    for k in K_DIAGNOSTIC_RANGE:
        labels_k = fcluster(Z, t=k, criterion="maxclust")
        sil_k = silhouette_score(X, labels_k, metric="euclidean")
        size_min = int(pd.Series(labels_k).value_counts().min())
        size_max = int(pd.Series(labels_k).value_counts().max())
        diag_rows.append({
            "k": k,
            "silhouette": round(float(sil_k), 4),
            "min_cluster_size": size_min,
            "max_cluster_size": size_max,
        })
        marker = "  <-- chosen" if k == K else ""
        print(f"  k={k}  silhouette={sil_k:+.4f}  sizes [{size_min}, {size_max}]{marker}")
    pd.DataFrame(diag_rows).to_csv(CLUSTER_DIAGNOSTICS_CSV, index=False)
    print(f"k-sweep diagnostic -> {CLUSTER_DIAGNOSTICS_CSV.name}")

    summary = (
        df.groupby("cluster")
          .agg(
              count=("cluster", "size"),
              mean_chf_per_m2=("chf_per_m2", "mean"),
              mean_rooms=("rooms", "mean"),
              dominant_canton=("canton", lambda s: s.value_counts().idxmax()),
          )
          .reset_index()
          .round({"mean_chf_per_m2": 2, "mean_rooms": 2})
    )
    summary.to_csv(CLUSTER_SUMMARY_CSV, index=False)
    print(f"\nCluster summary -> {CLUSTER_SUMMARY_CSV.name}")
    print(summary.to_string(index=False))

    # Dendrogram coloured with the shared palette.
    set_link_color_palette(PALETTE)
    fig, ax = plt.subplots(figsize=(12, 6))
    # color_threshold just below the merge height that separates k=6 from k=5,
    # so each of the top six sub-trees gets its own palette colour.
    ct = float(Z[-(K - 1), 2]) - 1e-9
    dendrogram(
        Z,
        truncate_mode="lastp",
        p=30,
        color_threshold=ct,
        above_threshold_color="#888888",
        ax=ax,
    )
    ax.set_title(f"Ward-linkage dendrogram (truncated, last 30 merges), k={K}")
    ax.set_xlabel("Cluster (sub-tree leaf count in parentheses)")
    ax.set_ylabel("Ward distance")
    fig.tight_layout()
    fig.savefig(DENDROGRAM_PNG, dpi=120)
    plt.close(fig)
    print(f"\nDendrogram -> {DENDROGRAM_PNG.name}")

    # Persist scaler + metadata. Step 4 (model.py) will extend this pickle
    # with per-cluster log-linear regressions and centroids.
    with open(MODEL_PKL, "wb") as f:
        pickle.dump(
            {
                "scaler": scaler,
                "features": FEATURES,
                "palette": PALETTE,
                "k": K,
            },
            f,
        )
    print(f"Persisted scaler + metadata -> {MODEL_PKL.name}")


if __name__ == "__main__":
    main()
