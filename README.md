# Customer Intelligence Engine — Team FT-US

**Home Credit · BI Season 10 · Round 2 (Synthesize)** · Leader: Trần Long Triều

An executive Streamlit dashboard that scores all **1,000,528 customers** on two axes — **Financial Health (FHS)** and **Engagement (ES)** — and turns the pair into exactly **one Next Best Action per customer**, each with a product, an offer amount, a channel, a timing and a plain-English reason.

---

## Run locally (about 2 minutes)

```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

The app opens at <http://localhost:8501>. (`python -m streamlit` works even when the Scripts folder is not on PATH.)

---

## The six pages

| # | Page | What it does |
|---|------|--------------|
| 01 | **Overview** | 5 KPI cards with delta pills, FHS × ES heat-map, sub-segment bars (**click a bar to filter the whole page**), segment donut, FHS distribution, NBA mix |
| 02 | **Financial Health** (C1) | Information Value of 18 factors, the five-tier risk ladder, an **interactive FHS scorecard what-if simulator** (waterfall from the 60-point base → tier + reason code), and an **interactive credit-appetite cut-off** showing volume kept vs bad loans avoided, by loans *and* by VND |
| 03 | **Engagement** (C2) | R-F-I-L-P by level, drivers vs reducers, PULL vs PUSH, and a **telesales call-budget optimizer** (cut 0–60 % of the lowest-ES calls) |
| 04 | **Segments & NBA** (C3) | FHS × ES matrix with the two risk guard-rails, the opportunity map highlighting *The Eager-Borrower Trap*, and a **campaign target-list exporter** with a live audit of the engine's hard rules and a CSV download |
| 05 | **Customer 360°** | Five quick-select personas or any `SKP_CLIENT`: two gauges, the five engagement dimensions, the recommended action and the verbatim `EXPLANATION` |
| 06 | **Insights & Decisions** | The four signature insights (I1 Pull vs Push, I2 Credit Hunger, I3 Maturity Window, I4 Ghost Calling), *Opportunity Cost — Both Ways*, and two A/B tests with a 10 % hold-out simulator |

The **Segment / FHS tier / ES level** slicers are shared across pages, and **🔄 Reset All Filters** returns to the full book in one click.

---

## Light-theme hard lock

The dashboard is pinned to a light report canvas regardless of the viewer's operating-system or browser colour scheme:

* every Plotly figure goes through `apply_pro_theme()` and is rendered with `st.plotly_chart(..., theme=None)`, so Streamlit's dark template can never repaint tick labels, axis titles or legends;
* all widget chrome (select boxes, drop-down menus, text inputs, multi-selects, sliders, tables) is pinned to a white surface with navy ink — covering **both** the BaseWeb DOM of older Streamlit and the React Aria DOM of newer Streamlit;
* `.streamlit/config.toml` locks the Streamlit theme itself to `base = "light"`.

Verified in an emulated dark-mode browser: axis ticks `#1E293B`, legends `#0F2043`, body text `#0F172A`, select boxes `#FFFFFF` — identical to light mode.

---

## Data the app reads

The app uses the first file it finds, next to `app.py` or in `./data`:

1. `customer_scores_nba.parquet` / `.csv` — the full output file (1,000,528 rows), **not** in this repo (too large for git).
2. `customer_scores_sample.parquet` or `customer_scores_sample.csv.gz` — **included here**: 50,000 real rows of the final run (5,000 per sub-segment), re-weighted back to the full base so portfolio estimates stay correct. The `.csv.gz` works without `pyarrow`.
3. No file at all → a 10,000-row synthetic demo set, so the app always opens.

A file can also be uploaded from the sidebar.

The sample carries 44 columns — `SKP_CLIENT`, `FHS`/`FHS_TIER`, `ES`/`ES_LEVEL`, `SEGMENT`/`SUB_SEGMENT`, the full `NBA_*` action block and `EXPLANATION` — plus a derived `WEIGHT` column added at load time. Customers are scored **as of 4 Jun 2025**.

To rebuild the sample from the full CSV:

```bash
python make_sample_for_cloud.py customer_scores_nba.csv
```

### Consistency with the notebook

Portfolio-level figures are the official full-base results, hard-coded in `OFFICIAL` in `app.py` and exported from `summary_metrics.json` of the final run (`2026-10-18 05:16:34`). All **831** of those values were diffed against the notebook's `summary_metrics.json` and match exactly; the bundled sample reproduces the official distributions to within **1.03 standard errors**.

---

## Deploy on Streamlit Community Cloud

1. Go to <https://share.streamlit.io> → **New app** → **Deploy a public app from GitHub**.
2. Repository `k642512255032-max/BI10-R02-FT-US-dashboard`, branch `main`, main file `app.py`.
3. **Deploy**. Dependencies install from `requirements.txt`.

---

## Repository layout

```
app.py                          the whole dashboard (single file)
requirements.txt                pinned to the tested stack
.streamlit/config.toml          locks Streamlit to the light theme
customer_scores_sample.parquet  50,000-row stratified sample (zstd)
customer_scores_sample.csv.gz   same sample, no pyarrow needed
make_sample_for_cloud.py        rebuilds the sample from the full CSV
```
