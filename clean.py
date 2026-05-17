"""
clean.py - sanitise data/raw.csv into data/clean.csv.

Drops rows missing price/size/plz, applies range filters
(size 15-500, price 300-15000, rooms 1-10), computes chf_per_m2,
then runs IQR outlier removal on chf_per_m2 (Q1 - 1.5*IQR, Q3 + 1.5*IQR).
Defensively drops rows with missing lat/lon or canton, and prints a
breakdown of which filter caught what.

pgeocode geocoding is skipped at this stage: scrape.py already supplies
real-address lat/lon from Flatfox at 100% coverage, which is more
precise than a PLZ centroid.

Usage:
    python clean.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Windows consoles default to cp1252 and choke on non-ASCII (→, -, etc.). Force
# stdout to UTF-8 so the summary prints regardless of the user's locale.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


DATA_DIR = Path(__file__).parent / "data"
RAW_CSV = DATA_DIR / "raw.csv"
CLEAN_CSV = DATA_DIR / "clean.csv"

PRICE_MIN, PRICE_MAX = 300, 15_000
SIZE_MIN, SIZE_MAX = 15, 500
ROOMS_MIN, ROOMS_MAX = 1, 10


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Apply the spec's filters sequentially and return (kept_df, drop_counts)."""
    drops: dict[str, object] = {"_started": len(df)}

    # 1. Drop rows missing the three required raw fields.
    mask = df["price_chf"].isna() | df["size_m2"].isna() | df["plz"].isna()
    drops["missing_price_size_plz"] = int(mask.sum())
    df = df[~mask]

    # 2. Price range - also kills the CHF 0 listings and any > 15k legitimately
    # over the rental ceiling.
    mask = (df["price_chf"] < PRICE_MIN) | (df["price_chf"] > PRICE_MAX)
    drops["price_out_of_range"] = int(mask.sum())
    df = df[~mask]

    # 3. Size range - kills the 0-m² entries and anything > 500 m² (mansion-grade).
    mask = (df["size_m2"] < SIZE_MIN) | (df["size_m2"] > SIZE_MAX)
    drops["size_out_of_range"] = int(mask.sum())
    df = df[~mask]

    # 4. Rooms range. Spec is `rooms ∈ [1, 10]`; NaN comparisons evaluate False
    # in pandas which would silently keep them, so we drop NaN explicitly.
    mask = df["rooms"].isna() | (df["rooms"] < ROOMS_MIN) | (df["rooms"] > ROOMS_MAX)
    drops["rooms_out_of_range_or_missing"] = int(mask.sum())
    df = df[~mask]

    # 5. Compute chf_per_m2. Use .copy() to silence pandas's SettingWithCopyWarning.
    df = df.copy()
    df["chf_per_m2"] = df["price_chf"] / df["size_m2"]

    # 6. IQR outlier removal on chf_per_m2. Computed AFTER the range filters so
    # the quartiles aren't pulled by the obvious bad rows.
    q1, q3 = df["chf_per_m2"].quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    mask = (df["chf_per_m2"] < lo) | (df["chf_per_m2"] > hi)
    drops["chf_per_m2_iqr_outlier"] = int(mask.sum())
    drops["_iqr_bounds"] = (round(float(lo), 2), round(float(hi), 2))
    df = df[~mask]

    # 7. Defensive lat/lon drop (scrape.py supplies 100% from Flatfox, so this
    # should be a no-op - but cluster.py needs lat/lon and we'd rather fail
    # closed than emit a row that breaks downstream).
    mask = df["latitude"].isna() | df["longitude"].isna()
    drops["missing_lat_lon"] = int(mask.sum())
    df = df[~mask]

    # 8. Canton must be set for cluster.py / app.py grouping.
    mask = df["canton"].fillna("").astype(str).str.strip() == ""
    drops["missing_canton"] = int(mask.sum())
    df = df[~mask]

    drops["_kept"] = len(df)
    return df.reset_index(drop=True), drops


def print_summary(drops: dict) -> None:
    started = drops["_started"]
    kept = drops["_kept"]
    print(f"Loaded {started} rows from {RAW_CSV.name}")
    print(f"Kept   {kept} rows ({kept * 100 / max(started, 1):.1f}%) → {CLEAN_CSV.name}")
    print()
    print("Drop reasons (sequential filters):")
    print(f"  missing price / size / plz             : {drops['missing_price_size_plz']:5}")
    print(f"  price not in [{PRICE_MIN}, {PRICE_MAX}]              : {drops['price_out_of_range']:5}")
    print(f"  size  not in [{SIZE_MIN}, {SIZE_MAX}]                    : {drops['size_out_of_range']:5}")
    print(f"  rooms not in [{ROOMS_MIN}, {ROOMS_MAX}] or missing        : {drops['rooms_out_of_range_or_missing']:5}")
    lo, hi = drops["_iqr_bounds"]
    print(f"  chf_per_m2 IQR outliers (bounds {lo} - {hi}) : {drops['chf_per_m2_iqr_outlier']:5}")
    print(f"  missing lat/lon                        : {drops['missing_lat_lon']:5}")
    print(f"  missing canton                         : {drops['missing_canton']:5}")


def main() -> None:
    if not RAW_CSV.exists():
        raise SystemExit(f"missing input: {RAW_CSV}. Run scrape.py first.")
    df = pd.read_csv(RAW_CSV)
    cleaned, drops = clean(df)
    cleaned.to_csv(CLEAN_CSV, index=False)

    print_summary(drops)
    print()
    print("chf_per_m2 stats on kept rows:")
    print(cleaned["chf_per_m2"].describe().round(2).to_string())
    print()
    print("canton distribution on kept rows (top 10):")
    print(cleaned["canton"].value_counts().head(10).to_string())


if __name__ == "__main__":
    main()
