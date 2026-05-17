"""
app.py - Streamlit UI for the Swiss Rent Fairness Analyser.

Four tabs:
  1. Market Map: folium + MarkerCluster, markers coloured by cluster.
  2. Am I Paying Too Much?: form -> evaluate_fairness, verdict banner,
     metrics, bar chart, ten nearest comparable listings.
  3. What Should I Charge?: form -> optimize_rent, metrics, histogram
     of cluster prices with a suggestion line.
  4. Method: pipeline writeup, embedded dendrogram, cluster summary table,
     paragraph on each ML choice.

Sidebar filters (canton, cluster, price range, rooms range) scope Tab 1
only. The evaluator and optimiser always use the full per-cluster
regressions so sidebar state doesn't silently affect predictions.

Caching: st.cache_data on the CSV loaders, st.cache_resource on the
FairPriceModel. The model is loaded once and never refit at request time.
On a fresh deploy data/model.pkl is missing (gitignored) so load_model
rebuilds it from data/clustered.csv on first request.
"""

from __future__ import annotations

from pathlib import Path

import folium
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium

from cluster import PALETTE
from model import FairPriceModel


# ----------------------------------------------------------------------------
# Page config + data loaders
# ----------------------------------------------------------------------------

st.set_page_config(
    page_title="Swiss Rent Fairness Analyser",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = Path(__file__).parent / "data"
CLUSTERED_CSV = DATA_DIR / "clustered.csv"
CLUSTER_SUMMARY_CSV = DATA_DIR / "cluster_summary.csv"
DENDROGRAM_PNG = DATA_DIR / "dendrogram.png"


@st.cache_data
def load_clustered() -> pd.DataFrame:
    return pd.read_csv(CLUSTERED_CSV)


@st.cache_data
def load_cluster_summary() -> pd.DataFrame:
    return pd.read_csv(CLUSTER_SUMMARY_CSV)


@st.cache_resource
def load_model() -> FairPriceModel:
    # data/model.pkl is gitignored, so a fresh Streamlit Cloud deploy starts
    # without it. data/clustered.csv IS tracked, so we can rebuild the pickle
    # in-place on the first cold start. Subsequent requests reuse the cache.
    pkl_path = DATA_DIR / "model.pkl"
    if not pkl_path.exists():
        from model import build as _rebuild_model
        _rebuild_model()
    return FairPriceModel()


df = load_clustered()
summary = load_cluster_summary()
model = load_model()


# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------

st.title("Swiss Rent Fairness Analyser")
st.caption(
    f"Hierarchical clustering of {len(df):,} Swiss apartments scraped from "
    "flatfox.ch, with per-cluster log-linear rent regression."
)


# ----------------------------------------------------------------------------
# Sidebar - filters affect Tab 1 (Market Map) only.
# ----------------------------------------------------------------------------

with st.sidebar:
    st.header("Map filters (Tab 1 only)")
    st.caption(
        "These filters scope the map markers. The evaluator and optimiser "
        "always use the full per-cluster models."
    )

    cantons = sorted(df["canton"].dropna().unique())
    selected_cantons = st.multiselect("Canton", cantons, default=cantons)

    cluster_options = sorted(df["cluster"].unique())
    selected_clusters = st.multiselect("Cluster", cluster_options, default=cluster_options)

    price_lo = int(df["price_chf"].min())
    price_hi = int(df["price_chf"].max())
    price_range = st.slider(
        "Monthly rent (CHF)", price_lo, price_hi, (price_lo, price_hi), step=100
    )

    rooms_lo = float(df["rooms"].min())
    rooms_hi = float(df["rooms"].max())
    rooms_range = st.slider(
        "Rooms", rooms_lo, rooms_hi, (rooms_lo, rooms_hi), step=0.5
    )


mask = (
    df["canton"].isin(selected_cantons)
    & df["cluster"].isin(selected_clusters)
    & df["price_chf"].between(*price_range)
    & df["rooms"].between(*rooms_range)
)
df_map = df[mask]


# ----------------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs(
    ["Market Map", "Am I Paying Too Much?", "What Should I Charge?", "Method"]
)


# --- Tab 1: Market Map -----------------------------------------------------

with tab1:
    st.subheader(f"Market Map - showing {len(df_map):,} of {len(df):,} listings")

    if len(df_map) == 0:
        st.warning("No listings match your filters. Widen the sidebar selection.")
    else:
        m = folium.Map(location=[46.948, 7.4474], zoom_start=8, tiles="cartodbpositron")
        mc = MarkerCluster().add_to(m)
        for _, row in df_map.iterrows():
            colour = PALETTE[int(row["cluster"]) - 1]
            popup = (
                f"<b>CHF {row['price_chf']:,.0f}/mo</b><br>"
                f"{row['rooms']:g} rooms · {row['size_m2']:.0f} m² · "
                f"{row['price_chf']/row['size_m2']:.1f} CHF/m²<br>"
                f"{row['city']}, {row['plz']}<br>"
                f"Cluster #{int(row['cluster'])}<br>"
                f"<a href='{row['url']}' target='_blank'>view listing</a>"
            )
            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=5,
                color=colour,
                fill=True,
                fill_color=colour,
                fill_opacity=0.75,
                weight=1,
                popup=folium.Popup(popup, max_width=300),
            ).add_to(mc)

        # Floating legend (HTML overlay rendered inside the map iframe).
        legend_rows = "".join(
            f'<span style="display:inline-block;width:12px;height:12px;'
            f'background:{PALETTE[int(r["cluster"]) - 1]};'
            f'margin-right:6px;border:1px solid #555;vertical-align:middle;"></span>'
            f'Cluster {int(r["cluster"])}: {r["mean_chf_per_m2"]:.1f} CHF/m² '
            f'(top: {r["dominant_canton"]})<br>'
            for _, r in summary.iterrows()
        )
        legend_html = (
            '<div style="position:fixed;top:80px;right:20px;z-index:9999;'
            'background:white;padding:10px 12px;border:1px solid #888;'
            'border-radius:4px;font-size:12px;box-shadow:0 2px 4px rgba(0,0,0,.15);">'
            "<b>Clusters</b><br>"
            f"{legend_rows}</div>"
        )
        m.get_root().html.add_child(folium.Element(legend_html))

        st_folium(m, height=620, returned_objects=[], use_container_width=True)


# --- Tab 2: Evaluate fairness ----------------------------------------------

VERDICT_STYLE = {
    "underpriced":          {"bg": "#22a884", "label": "Underpriced"},
    "fair":                 {"bg": "#22a884", "label": "Fair"},
    "slightly_overpriced":  {"bg": "#fb8c00", "label": "Slightly overpriced"},
    "overpriced":           {"bg": "#e53935", "label": "Overpriced"},
}


with tab2:
    st.subheader("Am I Paying Too Much?")
    st.caption(
        "Compares a listing against the log-linear regression for its "
        "geographically-nearest cluster. Reports a 95 % parametric "
        "**prediction interval** on log-residuals (PI, not CI)."
    )

    with st.form("eval_form"):
        c1, c2 = st.columns(2)
        with c1:
            price_in = st.number_input(
                "Monthly rent (CHF, gross)", min_value=300, max_value=15_000,
                value=2_000, step=50,
            )
            size_in = st.number_input(
                "Size (m²)", min_value=15, max_value=500, value=80, step=5,
            )
        with c2:
            rooms_in = st.number_input(
                "Rooms", min_value=1.0, max_value=10.0, value=3.0, step=0.5,
            )
            plz_in = st.text_input("Postal code (PLZ)", value="3000")
        submitted = st.form_submit_button("Evaluate", type="primary")

    if submitted:
        result = model.evaluate_fairness(
            float(price_in), float(size_in), float(rooms_in), plz_in
        )
        if "error" in result:
            st.error(result["error"])
        else:
            v = result["verdict"]
            style = VERDICT_STYLE[v]
            st.markdown(
                f'<div style="background:{style["bg"]};color:white;'
                f'padding:14px 18px;border-radius:6px;font-size:18px;'
                f'margin-bottom:12px;">'
                f'<b>{style["label"]}</b> - '
                f'actual is {result["delta_pct"]:+.1f}% vs expected'
                f"</div>",
                unsafe_allow_html=True,
            )

            c1, c2, c3 = st.columns(3)
            c1.metric("Actual", f"CHF {result['actual_chf']:,.0f}")
            c2.metric(
                "Expected (model)",
                f"CHF {result['expected_chf']:,.0f}",
                delta=f"{result['delta_pct']:+.1f}%",
                delta_color="off",
            )
            pi_lo, pi_hi = result["prediction_interval"]
            c3.metric("95 % PI", f"{pi_lo:,.0f} - {pi_hi:,.0f}")

            st.caption(
                f"Assigned cluster #{result['cluster_id']} "
                f"({result['n_comparables']:,} comparables). "
                f"PI = exp(log_pred ± 1.96 × log-residual_std), "
                f"computed from the cluster's held-out test set."
                + (" Cluster regression was sparse; global fallback used."
                   if result["is_fallback"] else "")
            )

            chart_df = pd.DataFrame({
                "type": ["Expected (model)", "Actual (listing)"],
                "CHF/month": [result["expected_chf"], result["actual_chf"]],
            })
            st.bar_chart(chart_df.set_index("type"), color="#414487")

            # Top-10 nearest comparables in the assigned cluster.
            in_lat = result["latitude"]
            in_lon = result["longitude"]
            cluster_df = df[df["cluster"] == result["cluster_id"]].copy()
            # Quick equirectangular distance in km - Switzerland is small so the
            # approximation (111 km / deg lat, 75 km / deg lon at 47°) is fine.
            cluster_df["dist_km"] = np.sqrt(
                ((cluster_df["latitude"] - in_lat) * 111.0) ** 2
                + ((cluster_df["longitude"] - in_lon) * 75.0) ** 2
            )
            top10 = (
                cluster_df.nsmallest(10, "dist_km")
                [["city", "plz", "rooms", "size_m2", "price_chf", "dist_km", "url"]]
                .round({"dist_km": 1, "size_m2": 0, "price_chf": 0})
                .rename(columns={
                    "city": "City", "plz": "PLZ", "rooms": "Rooms",
                    "size_m2": "m²", "price_chf": "CHF/mo",
                    "dist_km": "km", "url": "Link",
                })
            )
            st.subheader("Ten nearest comparables in the cluster")
            st.dataframe(
                top10,
                hide_index=True,
                use_container_width=True,
                column_config={"Link": st.column_config.LinkColumn("Link")},
            )


# --- Tab 3: Optimise rent --------------------------------------------------

with tab3:
    st.subheader("What Should I Charge?")
    st.caption(
        "Returns the cluster's empirical 10th / 90th log-residual percentile "
        "band - i.e. where comparable listings actually price."
    )

    with st.form("opt_form"):
        c1, c2 = st.columns(2)
        with c1:
            size_in_o = st.number_input(
                "Size (m²)", min_value=15, max_value=500, value=80, step=5,
                key="opt_size",
            )
        with c2:
            rooms_in_o = st.number_input(
                "Rooms", min_value=1.0, max_value=10.0, value=3.0, step=0.5,
                key="opt_rooms",
            )
        plz_in_o = st.text_input("Postal code (PLZ)", value="3000", key="opt_plz")
        submitted_o = st.form_submit_button("Suggest rent", type="primary")

    if submitted_o:
        result = model.optimize_rent(
            float(size_in_o), float(rooms_in_o), plz_in_o
        )
        if "error" in result:
            st.error(result["error"])
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Low (10th %)", f"CHF {result['low_chf']:,.0f}")
            c2.metric("Suggested", f"CHF {result['suggested_chf']:,.0f}")
            c3.metric("High (90th %)", f"CHF {result['high_chf']:,.0f}")
            st.caption(result["explanation"])

            cluster_df = df[df["cluster"] == result["cluster_id"]]
            fig = px.histogram(
                cluster_df, x="price_chf", nbins=40,
                title=(
                    f"Cluster #{result['cluster_id']} rent distribution "
                    f"(n = {len(cluster_df):,})"
                ),
                color_discrete_sequence=[PALETTE[result["cluster_id"] - 1]],
            )
            fig.add_vline(
                x=result["suggested_chf"],
                line_dash="dash", line_color="red",
                annotation_text=f"Suggested: CHF {result['suggested_chf']:,.0f}",
                annotation_position="top",
            )
            fig.add_vrect(
                x0=result["low_chf"], x1=result["high_chf"],
                fillcolor="red", opacity=0.08, line_width=0,
                annotation_text="10-90 % band", annotation_position="top left",
            )
            fig.update_layout(xaxis_title="Monthly rent (CHF)", yaxis_title="Listings")
            st.plotly_chart(fig, use_container_width=True)


# --- Tab 4: Method ---------------------------------------------------------

with tab4:
    st.subheader("Method")

    st.markdown(
        """
**Pipeline**

1. **Scrape** - Flatfox public JSON API (`flatfox.ch/api/v1/public-listing/`).
   The original homegate.ch target is fronted by DataDome CAPTCHA;
   Flatfox is the open alternative. APARTMENT-only filter, price reconstructed
   as gross rent (`rent_gross` or `rent_net + rent_charges`) for single
   consistent semantic. Real-address latitude/longitude come directly from
   Flatfox (more precise than PLZ-centroid geocoding).

2. **Clean** - IQR outlier removal on `chf_per_m2`, range filters on price /
   size / rooms, defensive null-drop on lat/lon and canton. Result: 5010 raw
   → 3428 cleaned apartments.

3. **Cluster** - Hierarchical clustering with Ward linkage on
   `(chf_per_m2, rooms, lat, lon)`, all StandardScaler-d. Cut at k=6 with
   `scipy.cluster.hierarchy.fcluster` (maxclust). Six chosen as a balance
   between interpretability and cluster balance - k=4 and k=5 give marginally
   higher silhouette but produce one dominant cluster (37-47 % of data) which
   would mechanically own most of the downstream regression.

4. **Per-cluster log-linear regression** - For each cluster, fit
   `log(price_chf) ~ log(size_m2) + rooms` on an 80 % training split,
   compute residual std + 10/90 percentiles on the held-out 20 % test set,
   all in log-space. Sparse-cluster fallback (n < 30) uses a globally fit
   regression instead; doesn't trigger on the current dataset.
        """
    )

    st.image(
        str(DENDROGRAM_PNG),
        caption="Ward-linkage dendrogram (truncated, last 30 merges, k=6)",
    )

    st.subheader("Cluster summary")
    st.dataframe(summary, hide_index=True, use_container_width=True)

    st.markdown(
        """
**Why these choices**

- **Hierarchical clustering over k-means**: deterministic (no random init),
  the dendrogram is a useful artifact to show, and there's no implicit
  spherical-cluster assumption.
- **Ward linkage**: minimises within-cluster variance, gives
  similarly-sized clusters. That matters because each cluster gets its
  own regression and I want each one to have a usable sample size.
- **Features `(chf_per_m2, rooms, lat, lon)`**: `chf_per_m2` captures the
  market tier, `rooms` captures the apartment archetype, `lat/lon` keeps
  comparables spatially coherent. All four standardised so no single
  feature dominates the Euclidean distance.
- **Log-linear regression**: rents are approximately log-normal -
  log-linear gives constant percentage error rather than constant CHF
  error, the right inductive bias for prices.
- **Train/test split for residuals**: residual std and 10/90 percentiles
  are computed on a held-out test set, not in-sample. In-sample
  residuals give optimistic intervals.

**Two uncertainty framings (deliberate)**

- **Evaluator** (Tab 2) → 95 % *parametric prediction interval* (PI),
  `exp(log_pred ± 1.96 · log-residual_std)`. Asks the statistical
  question: "is this listing within model uncertainty?" Assumes
  normal log-residuals.
- **Optimiser** (Tab 3) → *empirical* 10th/90th log-residual
  percentiles, `exp(log_pred + percentile)`. Asks the market question:
  "where do comparable listings actually price?"

Note on terminology: `±1.96 · σ` is a *prediction interval* (PI) for a
single new observation, not a *confidence interval* (CI). CI is for the
mean prediction; PI is for a single new observation. Different.

**Multicollinearity acknowledgment**: `log(size_m2)` and `rooms`
correlate ~0.7-0.85. We use the model for prediction only - coefficient
interpretation is not exposed - so multicollinearity is acknowledged
but does not threaten validity.

**Cluster assignment for new listings**: the clustering uses 4D features
including `chf_per_m2`, but `chf_per_m2` is unknown for a new listing
(it's what we're trying to predict). Inference assigns by nearest
centroid in the standardised `(rooms, lat, lon)` 3D subspace only.
Consequence: a listing's inference-time cluster can differ from its
training-time cluster. Expected, not a bug.

**Limitations / what production would need**

- *Cluster hybridity*: clusters mix geographic and price signal; two
  listings can land in the same cluster either because they're nearby OR
  because they have similar price/size.
- *No quality features*: year built, floor, balcony, condition, energy
  class - all missing from Flatfox. Material to price.
- *PLZ-centroid geocoding fallback at inference*: scrape.py uses
  real-address lat/lon, but new-listing input only has PLZ, geocoded to
  the PLZ centroid via pgeocode.
- *Point-in-time snapshot*: no re-scraping cron.
- *No baseline comparison*: a naïve "mean CHF/m² per canton" baseline
  would prove clustering adds value over a non-clustered approach. Not
  built here - acknowledged.
        """
    )
