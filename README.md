# Swiss Rent Fairness Analyser

> Streamlit app that scrapes ~5,000 Swiss apartment listings, clusters them into k=6 market tiers via Ward-linkage hierarchical clustering, and answers two questions:
> **"Am I paying too much?"** (for renters) and **"What should I charge?"** (for landlords).

**Live demo**: _(deploy URL added in Step 7)_

---

## Problem

Swiss apartment rents vary by an order of magnitude across cantons, neighbourhoods, and apartment archetypes. A renter looking at a single listing can't easily tell whether CHF 2,400/month for a 75 m² flat is fair, generous, or extortionate without expensive market research. A landlord faces the symmetric problem on the way in: priced too low and you lose income, too high and the flat sits empty.

This project builds a small end-to-end ML pipeline that turns a few thousand real listings into two interactive tools that answer those questions, with the uncertainty surfaced honestly (and with the right statistical name attached — "prediction interval", not "confidence interval").

## Data

- **Source**: [Flatfox.ch](https://flatfox.ch) public JSON API (`flatfox.ch/api/v1/public-listing/`). Plain HTTP, no anti-bot, ~35k active rental listings nationwide of which ~5,000 are APARTMENT-only.
- **Why not homegate.ch?** The original target. Homegate is fronted by DataDome CAPTCHA and returns HTTP 403 to headless browsers. The pivot is documented in the scrape commit message and in `scrape.py`'s docstring.
- **Price semantic**: `price_chf` is reconstructed **gross rent (Bruttomiete)** — `rent_gross` if set, else `rent_net + rent_charges` if both set, otherwise the row is dropped. Single consistent monthly-total semantic across all 5,010 rows. A CHF 20,000/month cap rejects sale prices that occasionally leak into the RENT feed.
- **Geolocation**: real-address latitude / longitude come directly from Flatfox (100% populated, ~1 m precision). Canton is derived from PLZ via `pgeocode`.

## Pipeline & ML approach

```
scrape.py     →  data/raw.csv        (5,010 apartments)
clean.py      →  data/clean.csv      (3,428 apartments after IQR + range filters)
cluster.py    →  data/clustered.csv          (clean + cluster label 1-6)
              →  data/dendrogram.png
              →  data/cluster_summary.csv
              →  data/cluster_diagnostics.csv (k=4..8 silhouette sweep)
              →  data/model.pkl     (fitted StandardScaler + metadata)
model.py      →  data/model.pkl     (extended with per-cluster regressions)
app.py        →  Streamlit UI       (consumes data/clustered.csv + data/model.pkl)
```

### Clustering — `cluster.py`

- **Features**: `chf_per_m2`, `rooms`, `latitude`, `longitude`, all StandardScaler-d so no single feature dominates the Euclidean distance.
- **Linkage = Ward**: minimises total within-cluster variance, well-behaved with Euclidean distance on standardised features, gives similarly-sized clusters (matters because each cluster gets its own regression downstream — we want all of them to have a usable sample).
- **k = 6** via `scipy.cluster.hierarchy.fcluster(..., criterion='maxclust')`. A k-sweep diagnostic at k ∈ {4, 5, 6, 7, 8} (saved to `data/cluster_diagnostics.csv`) shows silhouette varies by only ~0.02 across this range — the structural signal is intrinsically weak because we deliberately mix geographic and price dimensions. The deciding factor is cluster balance: k=4 / k=5 give marginally higher silhouette but at the cost of one cluster owning 37–47% of the data, which would mechanically dominate the per-cluster regression. k=6 is the smallest k that keeps max/min cluster size ratio under ~2.5×.

### Per-cluster log-linear regression — `model.py`

For each of the 6 clusters:

1. **80/20 train/test split** (`random_state=42`).
2. Fit `log(price_chf) ~ log(size_m2) + rooms` on the train half.
3. On the held-out test half, compute residual std and 10th/90th percentiles, all in log-space.

The form is **log-linear** because rents are approximately log-normal — log-linear gives constant percentage error, which is the right inductive bias for prices, not constant CHF error. Residual statistics are computed on the test set (not in-sample) so the prediction intervals are honest rather than optimistic.

Sparse-cluster fallback: clusters with n < 30 fall back to a globally fit regression. Doesn't trigger on the current dataset (smallest cluster has 351 rows) but the code path exists.

**Multicollinearity acknowledgment**: `log(size_m2)` and `rooms` correlate ~0.7–0.85 in each cluster. Used for prediction only — coefficient interpretation is not exposed — so multicollinearity is acknowledged but does not threaten validity.

### Two uncertainty framings (deliberate, distinct)

The evaluator and the optimiser answer different questions, so they report different uncertainty:

- **Evaluator** (Tab 2) returns a **95% parametric prediction interval (PI)**: `exp(log_pred ± 1.96 · log-residual_std)`. Assumes normal log-residuals. Answers the statistical question: *"is this listing within model uncertainty?"*
- **Optimiser** (Tab 3) returns the **empirical 10th / 90th log-residual percentiles**, exponentiated back to CHF. Non-parametric. Answers the market question: *"where do comparable listings actually price?"*

On terminology — `±1.96 · σ` is a **prediction interval (PI)**, not a confidence interval (CI). CI is for the mean prediction; PI is for a single new observation. They are different and the code, comments, and UI all use the correct one.

### New-listing cluster assignment

The clustering uses 4D features including `chf_per_m2`. But for a new listing (renter/landlord input), `chf_per_m2` is unknown — it's what the evaluator is trying to predict. Inference assigns by nearest centroid in the StandardScaler-d **`(rooms, lat, lon)` 3D subspace** only. Consequence: a listing's inference-time cluster can differ from its training-time cluster. Expected, not a bug — documented in `model.py` and the Method tab.

## Running locally

```bash
python -m venv .venv
.venv\Scripts\activate                     # or source .venv/bin/activate
pip install -r requirements.txt
# optional: re-scrape from Flatfox
#   pip install -r requirements-scraper.txt
#   python scrape.py                       # ~7 min, fetches ~5,000 apartments
#   python clean.py
python cluster.py                          # generates data/clustered.csv + dendrogram + diagnostics
python model.py                            # rebuilds data/model.pkl (auto-rebuilt by app.py if missing)
streamlit run app.py                       # opens at http://localhost:8501
```

`data/clean.csv` and `data/clustered.csv` are tracked in the repo so the app runs without re-scraping (Flatfox content drifts day to day). Re-scraping is local-only — the scraper deps live in `requirements-scraper.txt`, separate from the Streamlit Cloud runtime in `requirements.txt`.

## Deployment notes (Streamlit Cloud)

1. Push the repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io), connect the repo, set main file to `app.py`.
3. First cold start takes ~30 seconds (pip install) + ~10 seconds (model.pkl rebuild from clustered.csv via `model.build()` called from `app.load_model`). After that, requests are sub-second.

`data/model.pkl` is gitignored — it's regenerated on first request from `data/clustered.csv` (tracked). Streamlit Cloud's container is read/write in the app directory, so this works.

## Limitations and what production would need

This is a deliberately small project. To be a real product it would need:

- **Cluster hybridity**: the clusters mix geographic and price signal. Two listings can land in the same cluster either because they're nearby or because they have similar price/size. Surfacing this honestly to users would matter.
- **Quality features**: year built, floor, balcony, condition, energy class, view — all missing from Flatfox and all materially affect price. Adding these would likely lift R² by 0.1–0.2 and tighten prediction intervals.
- **Address-level geocoding at inference**: `scrape.py` uses real-address lat/lon from Flatfox, but inference geocodes the user's PLZ via pgeocode (PLZ centroid only). For a real product the input would be a street address.
- **Time-of-listing dynamics**: this is a point-in-time snapshot. A production system would need a re-scrape cron and a temporal model (rents in Zurich rose ~6% year-on-year in 2024).
- **Baseline comparison**: a naïve "mean CHF/m² per canton" baseline would prove that clustering + log-linear regression actually adds value over a constant-per-canton prediction. Not built here — acknowledged.
- **Calibration check**: 95% PI should empirically contain ~95% of held-out listings. Coverage was not measured.
- **Live update**: re-scraping cron + automated rebuild of clustered.csv and model.pkl.

## Repo layout

```
housing-analyser/
├── README.md                  this file
├── CLAUDE.md                  project conventions (ML choices, git policy)
├── requirements.txt           runtime (no Playwright)
├── requirements-scraper.txt   scraper-only (stub — scraping uses stdlib urllib)
├── .gitignore
├── scrape.py                  Flatfox API scraper
├── clean.py                   IQR + range filters
├── cluster.py                 Ward hierarchical clustering, k=6
├── model.py                   per-cluster log-linear regression
├── app.py                     Streamlit UI
└── data/
    ├── clean.csv              tracked — input to cluster.py
    ├── clustered.csv          tracked — input to model.py + app.py
    ├── dendrogram.png         tracked — embedded in Method tab
    ├── cluster_summary.csv    tracked — embedded in Method tab
    ├── cluster_diagnostics.csv  tracked — k=4..8 silhouette sweep
    ├── raw.csv                gitignored — regenerable from scrape.py
    └── model.pkl              gitignored — regenerable from model.py
```

## License

This is a course project. No formal license; the data scraped from Flatfox is subject to their terms of use.
