# Swiss Rent Fairness Analyser — project conventions

Streamlit web app that scrapes Swiss rental listings (homegate.ch), clusters them hierarchically into ~6 price/region tiers, and surfaces three tools: market map, "am I paying too much?" evaluator, "what should I charge?" optimiser.

## Pipeline order
`scrape.py` → `data/raw.csv` → `clean.py` → `data/clean.csv` → `cluster.py` → `data/clustered.csv` + `data/dendrogram.png` + `data/cluster_summary.csv` → `model.py` → `data/model.pkl` → `app.py` (Streamlit).

## Data source (deviation from original spec)
Original target was homegate.ch via Playwright + __NEXT_DATA__ parsing. Homegate is fronted by **DataDome CAPTCHA** and returns 403 to headless Chromium. Pivoted to **Flatfox** (`flatfox.ch/api/v1/public-listing/`) — public JSON API, no anti-bot, stdlib urllib only. Trade-offs:
- Only ~5% of Flatfox listings are APARTMENT; total pool ≈ 1,700 apartments (below the spec's 3,000 target — documented in README limitations).
- Flatfox has no canton field → derived from PLZ via pgeocode.
- Flatfox provides exact-address `latitude`/`longitude` directly, so scrape.py emits them and clean.py uses them as-is (no PLZ-centroid pgeocode geocoding).
- `price_chf` is **reconstructed gross rent** (Bruttomiete): `rent_gross` if set, else `rent_net + rent_charges` if both set, else the row is dropped. Single consistent semantic for clustering.

## ML conventions (defensible — this is a marked project)
- **Prediction interval (PI), not confidence interval (CI).** `±1.96·σ` around a single new observation is a PI; CI is for the mean prediction.
- Per-cluster regression is **log-linear**: `log(price_chf) ~ log(size_m2) + rooms`, predict in log-space, `exp()` back to CHF.
- Residual std and 10/90 percentiles are computed on a **held-out test set** (80/20, `random_state=42`), never in-sample.
- New-listing cluster assignment uses only `(rooms, lat, lon)` standardised — `chf_per_m2` is unknown for a new listing.
- Sparse-cluster fallback: clusters with `n < 30` defer to a global log-linear regression.
- Evaluator returns parametric PI (assumes normal log-residuals). Optimiser returns empirical 10/90 percentiles. Two framings, deliberately distinct.

## Model caching
- `data/model.pkl` is built once by `model.py` and loaded by `app.py` via `st.cache_resource`. Never refit during a Streamlit interaction.

## Environment
- Python 3.12, fresh `.venv/` inside the project.
- Two requirements files:
  - `requirements.txt` — runtime (what Streamlit Cloud installs). No Playwright.
  - `requirements-scraper.txt` — scraping only. `playwright install chromium` after install.

## Git
- One descriptive commit per build step. **Never squash, never amend** — the commit history is part of the project defence.

## Build is step-gated
Build one file/step at a time, sanity-test, stop and confirm with the user before continuing.
