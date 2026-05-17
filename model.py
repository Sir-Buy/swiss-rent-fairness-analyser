"""
model.py — fair-price evaluator + rent optimiser (Step 4, the ML core part 2).

What this module exposes:

  evaluate_fairness(price_chf, size_m2, rooms, plz) -> dict
      For renters. Geocodes the PLZ, assigns the nearest cluster, runs
      that cluster's log-linear regression to get an expected gross rent,
      and reports a verdict + 95 % parametric *prediction interval*
      (PI, not CI — see "ML conventions" in CLAUDE.md).

  optimize_rent(size_m2, rooms, plz) -> dict
      For landlords. Same cluster assignment, returns suggested rent +
      empirical 10/90 percentile band from the cluster's test-set
      residuals.

Why two different uncertainty framings (in case anyone asks):
  - Evaluator: parametric Gaussian PI on log-residuals → "is this
    listing within model uncertainty?" Asks a statistical question.
  - Optimiser: empirical 10/90 of observed residuals → "where do
    similar listings actually price?" Asks a market question.
  They answer different questions on purpose.

Build (when run as a script):
  - Reads data/clustered.csv (3428 rows, 6 clusters)
  - For each cluster: 80/20 train/test split (random_state=42), fits
    log(price_chf) ~ log(size_m2) + rooms on train, computes residual std
    and 10/90 percentiles on test — both in log-space.
  - Cluster centroid in the StandardScaler-d (rooms, lat, lon) subspace
    (chf_per_m2 dimension dropped — unknown for a new listing).
  - Sparse-cluster fallback: n < SPARSE_THRESHOLD → reuse a globally fit
    regression instead. Won't trigger on the current dataset (smallest
    cluster is 351) but the code path exists and is exercised by the
    fallback unit test.
  - Saves everything into data/model.pkl (extending the Step 3 pickle).

Multicollinearity disclosure: log(size_m2) and rooms correlate ~0.7-0.85.
We use the regression for prediction only — coefficient interpretation
is not used downstream — so multicollinearity is acknowledged but does
not threaten the validity of the predictions or intervals.

Usage:
    python model.py            # rebuild data/model.pkl from clustered.csv
    python model.py --sanity   # rebuild and run a sanity check
"""

from __future__ import annotations

import argparse
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pgeocode
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


DATA_DIR = Path(__file__).parent / "data"
CLUSTERED_CSV = DATA_DIR / "clustered.csv"
MODEL_PKL = DATA_DIR / "model.pkl"

# A train/test residual analysis below SPARSE_THRESHOLD listings is unreliable
# (n=30 gives ~24 train / 6 test, with the test set barely supporting
# 10th/90th percentile estimation). We fall back to a global regression.
SPARSE_THRESHOLD = 30
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Z-score margin used for the parametric prediction interval in log-space.
PI_Z = 1.96  # ≈ 95 % under normal log-residuals

# Verdict thresholds — % deviation of actual from expected.
VERDICT_THRESHOLDS = [
    (-10.0, "underpriced"),     # actual < expected − 10 %
    (10.0, "fair"),             # within ± 10 %
    (25.0, "slightly_overpriced"),
    (float("inf"), "overpriced"),
]


# ----------------------------------------------------------------------------
# Internals
# ----------------------------------------------------------------------------

@dataclass
class ClusterModel:
    regression: LinearRegression
    residual_std: float        # log-space, computed on test set
    residual_p10: float        # log-space, computed on test set
    residual_p90: float        # log-space, computed on test set
    n_train: int
    n_test: int
    centroid_3d: np.ndarray    # standardised (rooms, lat, lon)
    is_fallback: bool          # True if this cluster used the global model

    @property
    def n_comparables(self) -> int:
        return self.n_train + self.n_test


# Pickle portability: when model.py is run as `python model.py`, this dataclass
# is defined under __module__ == "__main__". The resulting pickle would then
# refer to `__main__.ClusterModel`, which app.py (a different __main__) can't
# resolve. Two-step workaround:
#   1) Alias the running module as "model" in sys.modules. setdefault keeps the
#      no-op behaviour when model.py is normally imported as `model`.
#   2) Pin ClusterModel.__module__ to "model" so pickle dumps "model.ClusterModel".
# Both together: pickle's identity check (`sys.modules['model'].ClusterModel is
# ClusterModel`) passes regardless of build entry point.
sys.modules.setdefault("model", sys.modules[__name__])
ClusterModel.__module__ = "model"


def _fit_one(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[LinearRegression, float, float, float]:
    """Fit log(price) ~ log(size_m2) + rooms on train; return model + test-set residual stats."""
    Xtr = np.column_stack([np.log(train_df["size_m2"]), train_df["rooms"]])
    ytr = np.log(train_df["price_chf"])
    reg = LinearRegression().fit(Xtr, ytr)

    Xte = np.column_stack([np.log(test_df["size_m2"]), test_df["rooms"]])
    yte = np.log(test_df["price_chf"])
    pred = reg.predict(Xte)
    resid = yte - pred  # actual − predicted, log-space

    return reg, float(resid.std(ddof=1)), float(np.quantile(resid, 0.10)), float(np.quantile(resid, 0.90))


def _standardize_3d(scaler, rooms: float, lat: float, lon: float) -> np.ndarray:
    """Map (rooms, lat, lon) into the 3D subspace of the 4-feature StandardScaler.

    The Step-3 scaler was fit on (chf_per_m2, rooms, lat, lon). For a new
    listing we don't know chf_per_m2, so we only standardise the remaining
    three dimensions using the scaler's stored means / scales (indices 1-3).
    """
    means = scaler.mean_[1:]
    scales = scaler.scale_[1:]
    return (np.array([rooms, lat, lon], dtype=float) - means) / scales


def _verdict(delta_pct: float) -> str:
    if delta_pct < -10.0:
        return "underpriced"
    if delta_pct <= 10.0:
        return "fair"
    if delta_pct <= 25.0:
        return "slightly_overpriced"
    return "overpriced"


# ----------------------------------------------------------------------------
# Model build (run as script)
# ----------------------------------------------------------------------------

def build_models(df: pd.DataFrame, scaler) -> dict:
    """Returns the dict that ends up pickled (extends Step 3's pickle)."""
    print(f"Building per-cluster models from {len(df)} rows...")

    # Global fallback model — fit on the whole dataset, used when a cluster
    # is below SPARSE_THRESHOLD. Also serves as a "default" baseline.
    g_train, g_test = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    g_reg, g_std, g_p10, g_p90 = _fit_one(g_train, g_test)
    print(f"  global fallback   n={len(df):4} (train={len(g_train)}, test={len(g_test)}, "
          f"resid_std={g_std:.4f}, p10={g_p10:.4f}, p90={g_p90:.4f})")

    cluster_models: dict[int, ClusterModel] = {}
    for cid, grp in df.groupby("cluster"):
        # Centroid in the standardised (rooms, lat, lon) 3D subspace.
        means = scaler.mean_[1:]
        scales = scaler.scale_[1:]
        cent_raw = grp[["rooms", "latitude", "longitude"]].to_numpy(dtype=float)
        cent_3d = ((cent_raw - means) / scales).mean(axis=0)

        if len(grp) < SPARSE_THRESHOLD:
            cluster_models[int(cid)] = ClusterModel(
                regression=g_reg, residual_std=g_std,
                residual_p10=g_p10, residual_p90=g_p90,
                n_train=len(g_train), n_test=len(g_test),
                centroid_3d=cent_3d, is_fallback=True,
            )
            print(f"  cluster {cid}  n={len(grp):4}  -> SPARSE, using global fallback")
        else:
            tr, te = train_test_split(grp, test_size=TEST_SIZE, random_state=RANDOM_STATE)
            reg, std, p10, p90 = _fit_one(tr, te)
            cluster_models[int(cid)] = ClusterModel(
                regression=reg, residual_std=std,
                residual_p10=p10, residual_p90=p90,
                n_train=len(tr), n_test=len(te),
                centroid_3d=cent_3d, is_fallback=False,
            )
            print(f"  cluster {cid}  n={len(grp):4}  train={len(tr):4}  test={len(te):3}  "
                  f"resid_std={std:.4f}  p10={p10:+.4f}  p90={p90:+.4f}")

    return {
        "cluster_models": cluster_models,
        "global_model": ClusterModel(
            regression=g_reg, residual_std=g_std,
            residual_p10=g_p10, residual_p90=g_p90,
            n_train=len(g_train), n_test=len(g_test),
            centroid_3d=np.zeros(3), is_fallback=True,
        ),
    }


def build() -> None:
    """Build data/model.pkl from data/clustered.csv. Self-contained — does NOT
    depend on a pre-existing pickle, so this works on a fresh Streamlit Cloud
    deploy where cluster.py hasn't been re-run."""
    if not CLUSTERED_CSV.exists():
        raise SystemExit(f"missing input: {CLUSTERED_CSV}. Run cluster.py first.")

    # PALETTE/K/FEATURES live in cluster.py (single source of truth for the
    # clustering hyperparameters). Imported here at build time so we don't
    # have to depend on the Step 3 pickle existing.
    from cluster import FEATURES as CLUSTER_FEATURES, PALETTE, K

    df = pd.read_csv(CLUSTERED_CSV)
    # Refit StandardScaler from the clustered data. Identical to cluster.py's
    # scaler because both fit on the same rows + features. Refitting here
    # decouples model.build() from cluster.py's pickle.
    scaler = StandardScaler().fit(df[CLUSTER_FEATURES].to_numpy(dtype=float))

    extra = build_models(df, scaler)

    out = {
        "scaler": scaler,
        "features": CLUSTER_FEATURES,
        "palette": PALETTE,
        "k": K,
        **extra,
    }
    with open(MODEL_PKL, "wb") as f:
        pickle.dump(out, f)
    print(f"\nWrote {MODEL_PKL.name} (scaler + cluster_models + global_model).")


# ----------------------------------------------------------------------------
# Inference API (used by app.py)
# ----------------------------------------------------------------------------

class FairPriceModel:
    """Single-shot model loader + inference. Cache via st.cache_resource."""

    def __init__(self, path: Path = MODEL_PKL) -> None:
        with open(path, "rb") as f:
            d = pickle.load(f)
        self.scaler = d["scaler"]
        self.features = d["features"]
        self.palette = d["palette"]
        self.k: int = d["k"]
        self.cluster_models: dict[int, ClusterModel] = d["cluster_models"]
        self.global_model: ClusterModel = d["global_model"]
        self._geocoder = pgeocode.Nominatim("ch")

    def geocode_plz(self, plz: str | int) -> tuple[float, float] | None:
        plz_str = str(plz).strip().zfill(4)
        row = self._geocoder.query_postal_code(plz_str)
        lat, lon = row.get("latitude"), row.get("longitude")
        if pd.isna(lat) or pd.isna(lon):
            return None
        return float(lat), float(lon)

    def assign_cluster(self, rooms: float, lat: float, lon: float) -> int:
        new_3d = _standardize_3d(self.scaler, rooms, lat, lon)
        best_id, best_d = None, float("inf")
        for cid, cm in self.cluster_models.items():
            d = float(np.linalg.norm(new_3d - cm.centroid_3d))
            if d < best_d:
                best_id, best_d = cid, d
        assert best_id is not None
        return best_id

    def _predict_log(self, cm: ClusterModel, size_m2: float, rooms: float) -> float:
        X_new = np.array([[np.log(size_m2), rooms]])
        return float(cm.regression.predict(X_new)[0])

    def evaluate_fairness(self, price_chf: float, size_m2: float, rooms: float, plz: str | int) -> dict:
        """Renter-facing: is this listing fairly priced?

        Returns a dict with expected_chf, actual_chf, delta_pct, verdict,
        a 95 % *prediction interval* (PI, not CI), cluster_id, n_comparables,
        and an is_fallback flag indicating whether the global fallback model
        was used for the assigned cluster.
        """
        geo = self.geocode_plz(plz)
        if geo is None:
            return {"error": f"could not geocode PLZ {plz!r}"}
        lat, lon = geo

        cid = self.assign_cluster(rooms, lat, lon)
        cm = self.cluster_models[cid]
        log_pred = self._predict_log(cm, size_m2, rooms)
        expected_chf = float(np.exp(log_pred))
        delta_pct = (price_chf / expected_chf - 1.0) * 100.0
        half_width = PI_Z * cm.residual_std  # log-space
        pi_low = float(np.exp(log_pred - half_width))
        pi_high = float(np.exp(log_pred + half_width))

        return {
            "expected_chf": round(expected_chf, 2),
            "actual_chf": float(price_chf),
            "delta_pct": round(delta_pct, 2),
            "verdict": _verdict(delta_pct),
            "prediction_interval": (round(pi_low, 2), round(pi_high, 2)),
            "cluster_id": int(cid),
            "n_comparables": cm.n_comparables,
            "is_fallback": bool(cm.is_fallback),
            "latitude": lat,
            "longitude": lon,
        }

    def optimize_rent(self, size_m2: float, rooms: float, plz: str | int) -> dict:
        """Landlord-facing: what should I charge?

        Returns suggested_chf (point estimate), low_chf / high_chf from the
        cluster's empirical 10th / 90th log-residual percentiles, plus an
        explanation string.
        """
        geo = self.geocode_plz(plz)
        if geo is None:
            return {"error": f"could not geocode PLZ {plz!r}"}
        lat, lon = geo

        cid = self.assign_cluster(rooms, lat, lon)
        cm = self.cluster_models[cid]
        log_pred = self._predict_log(cm, size_m2, rooms)
        suggested = float(np.exp(log_pred))
        low = float(np.exp(log_pred + cm.residual_p10))
        high = float(np.exp(log_pred + cm.residual_p90))

        return {
            "suggested_chf": round(suggested, 2),
            "low_chf": round(low, 2),
            "high_chf": round(high, 2),
            "cluster_id": int(cid),
            "n_comparables": cm.n_comparables,
            "is_fallback": bool(cm.is_fallback),
            "explanation": (
                f"Based on {cm.n_comparables} comparable listings in your "
                f"cluster (cluster #{cid}), 80 % of similar apartments rent "
                f"between CHF {low:,.0f} and CHF {high:,.0f}. The point "
                f"estimate is the cluster's log-linear regression prediction."
            ),
            "latitude": lat,
            "longitude": lon,
        }


# Module-level convenience wrappers — useful for ad-hoc scripts / tests / app.
_MODEL: FairPriceModel | None = None


def _model() -> FairPriceModel:
    global _MODEL
    if _MODEL is None:
        _MODEL = FairPriceModel()
    return _MODEL


def evaluate_fairness(price_chf: float, size_m2: float, rooms: float, plz: str | int) -> dict:
    return _model().evaluate_fairness(price_chf, size_m2, rooms, plz)


def optimize_rent(size_m2: float, rooms: float, plz: str | int) -> dict:
    return _model().optimize_rent(size_m2, rooms, plz)


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def _sanity_check() -> None:
    """Pull a known row from clustered.csv, evaluate it, sanity-check predictions."""
    df = pd.read_csv(CLUSTERED_CSV)
    # Pick a row near the median chf_per_m2 of its cluster — most "average".
    df = df.assign(_chfm2=df["price_chf"] / df["size_m2"])
    df = df.sort_values("_chfm2").reset_index(drop=True)
    sample = df.iloc[len(df) // 2]

    print(f"\n--- sanity check ---")
    print(f"Sample: id={sample['id']}, {sample['city']} {sample['plz']}, "
          f"{sample['rooms']} rooms, {sample['size_m2']:.0f} m², "
          f"price CHF {sample['price_chf']:.0f}, real cluster {sample['cluster']}")

    m = FairPriceModel()
    eva = m.evaluate_fairness(
        price_chf=float(sample["price_chf"]),
        size_m2=float(sample["size_m2"]),
        rooms=float(sample["rooms"]),
        plz=str(sample["plz"]),
    )
    print(f"evaluate_fairness -> {eva}")
    if "error" in eva:
        return
    print(f"  delta_pct={eva['delta_pct']:.1f}%  (expected within +/- ~30 % for a typical listing)")
    print(f"  assigned cluster {eva['cluster_id']} (data row was cluster {sample['cluster']})")

    opt = m.optimize_rent(
        size_m2=float(sample["size_m2"]),
        rooms=float(sample["rooms"]),
        plz=str(sample["plz"]),
    )
    print(f"\noptimize_rent  -> suggested={opt['suggested_chf']}, "
          f"range=[{opt['low_chf']}, {opt['high_chf']}]")
    print(f"  {opt['explanation']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sanity", action="store_true", help="run a sanity check after building")
    args = parser.parse_args()

    build()
    if args.sanity:
        _sanity_check()


if __name__ == "__main__":
    main()
