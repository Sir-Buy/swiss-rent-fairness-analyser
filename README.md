# Swiss Rent Fairness Analyser

A Streamlit app for analysing Swiss apartment rents. Pipeline: scrape ~5,000 active rental listings, clean and filter them, group them into 6 market tiers with hierarchical clustering, then fit a per-cluster log-linear regression to predict rent.

**Live demo**: https://swiss-rent.streamlit.app

## Problem

Swiss apartment rents differ by an order of magnitude across cantons. A renter looking at one listing can't easily say whether CHF 2,400/month for a 75 m² flat is fair without expensive market research. A landlord has the same problem in reverse: too low and they lose income, too high and the flat sits empty. This project takes a few thousand real listings and turns them into two tools: a fairness check for renters and a rent suggestion for landlords.

## Data

- Source: the public JSON endpoint at `flatfox.ch/api/v1/public-listing/`. About 35,000 active rental listings nationally, of which roughly 5% are APARTMENT (the rest are parking, commercial, etc.).
- Original target was homegate.ch but it returns 403 to headless browsers behind a DataDome CAPTCHA. Pivoted to Flatfox.
- `price_chf` is reconstructed gross rent: `rent_gross` if set, otherwise `rent_net + rent_charges`. Rows missing both are dropped. A CHF 20,000/month cap catches the occasional sale price that leaks into the rental feed.
- Latitude and longitude come from Flatfox directly (real address, not a postal-code centroid). Canton is derived from PLZ via `pgeocode`.

## Pipeline

```
scrape.py     →  data/raw.csv               (5,010 apartments)
clean.py      →  data/clean.csv             (3,428 after filters)
cluster.py    →  data/clustered.csv
              →  data/dendrogram.png
              →  data/cluster_summary.csv
              →  data/cluster_diagnostics.csv
model.py      →  data/model.pkl             (per-cluster regressions)
app.py        →  Streamlit UI
```

### Clustering

Features: `chf_per_m2`, `rooms`, `latitude`, `longitude`. All standardised so no single feature dominates the Euclidean distance.

Linkage is Ward. Ward minimises within-cluster variance, is deterministic (no random initialisation), and gives similarly-sized clusters. Cluster balance matters here because each cluster gets its own regression downstream - a few oversize clusters would mechanically own most of the predictions.

k=6 was picked from a silhouette sweep over k=4..8 (saved to `data/cluster_diagnostics.csv`). Silhouette only varies by about 0.02 across that range, which means the clustering structure is intrinsically weak. That is expected when mixing geographic and price dimensions on purpose; the goal is interpretable market tiers, not maximally compact clusters. The deciding factor is cluster balance: k=4 and k=5 give marginally higher silhouette but produce one cluster at 37-47% of the data. k=6 is the smallest k that keeps the max/min size ratio below about 2.5.

### Per-cluster regression

For each cluster:

1. 80/20 train/test split, `random_state=42`.
2. Fit `log(price_chf) ~ log(size_m2) + rooms` on train.
3. On the held-out test split, compute residual standard deviation and 10th/90th percentiles, in log-space.

The form is log-linear because rents are approximately log-normal. Log-linear gives constant percentage error, which is the right inductive bias for prices, not constant CHF error. Residual statistics are computed on the test set (not in-sample) so the intervals are honest rather than optimistic.

Sparse-cluster fallback: clusters with n<30 fall back to a globally-fit regression. Does not trigger on this dataset (smallest cluster has 351 rows) but the code path exists.

Multicollinearity: `log(size_m2)` and `rooms` correlate around 0.7-0.85 inside each cluster. The regression is used for prediction only, not coefficient interpretation, so this is acknowledged but does not threaten validity.

### Two uncertainty framings

The evaluator and the optimiser answer different questions, so they report different uncertainty.

- **Evaluator** (Tab 2) returns a 95% **prediction interval**: `exp(log_pred ± 1.96 · log_residual_std)`. Parametric, assumes normal log-residuals. Answers: is this listing within model uncertainty?
- **Optimiser** (Tab 3) returns the cluster's empirical 10th/90th log-residual percentiles, exponentiated back to CHF. Non-parametric. Answers: where do comparable listings actually price?

Terminology: `±1.96·σ` around a single new observation is a *prediction interval* (PI), not a confidence interval. CI is for the mean prediction; PI is for a single new observation. Different objects, different names.

### New-listing cluster assignment

The clustering uses 4D features including `chf_per_m2`. For a new listing, `chf_per_m2` is the unknown - it is what the evaluator is trying to predict. So inference uses only the (rooms, lat, lon) subspace, standardised the same way, and picks the nearest cluster centroid. A listing's inference-time cluster can therefore differ from what its training-time cluster would have been. This is expected, and documented in `model.py` and the Method tab.

## Running locally

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python cluster.py
python model.py
streamlit run app.py
```

`data/clean.csv` and `data/clustered.csv` are committed so the app runs without re-scraping. To re-scrape from Flatfox:

```
pip install -r requirements-scraper.txt
python scrape.py
python clean.py
```

## Deployment

Deployed on Streamlit Cloud at https://swiss-rent.streamlit.app. The Python runtime is pinned to 3.12 via `.python-version` and `runtime.txt`. `data/model.pkl` is gitignored and is rebuilt from `clustered.csv` on first request (about 10 seconds of cold-start, then cached).

## Limitations

- Clusters mix geographic and price signal. Two listings can group together either because they are near each other or because they have similar price/size.
- No quality features. Year built, floor, balcony, condition, energy class are all missing from Flatfox and all matter for price.
- At inference, the user's PLZ is geocoded to a PLZ centroid via pgeocode, not the actual address.
- Point-in-time snapshot. No re-scraping cron.
- No baseline comparison. A "mean CHF/m² per canton" baseline would show that clustering plus regression actually beats a constant-per-canton estimate. Not built here.
- Empirical coverage of the 95% PI was not checked against held-out data.

## Repo layout

```
housing-analyser/
├── README.md
├── requirements.txt           runtime (no Playwright)
├── requirements-scraper.txt   scraper-only
├── runtime.txt                pins Python 3.12 for the deploy
├── .python-version
├── .gitignore
├── scrape.py                  Flatfox API scraper
├── clean.py                   range filters and IQR outlier removal
├── cluster.py                 Ward hierarchical clustering, k=6
├── model.py                   per-cluster log-linear regression
├── app.py                     Streamlit UI
└── data/
    ├── clean.csv              tracked
    ├── clustered.csv          tracked
    ├── dendrogram.png         tracked
    ├── cluster_summary.csv    tracked
    ├── cluster_diagnostics.csv
    ├── raw.csv                gitignored
    └── model.pkl              gitignored, rebuilt at runtime
```
