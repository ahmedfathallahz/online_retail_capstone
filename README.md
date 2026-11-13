# Online Retail Customer Analytics & Segmentation

End-to-end analysis of a UK-based online gift retailer’s transaction data (Dec 2010 – Dec 2011).  
The work covers data cleaning, exploratory analysis, RFM scoring, customer segmentation, and an interactive dashboard.

---

## Project Structure

```
online_retail_capstone/
├── data/
│   ├── Online_Retail.xlsx
│   ├── registered_clean.csv
│   ├── guests.csv
│   ├── cancellations.csv
│   ├── anomalies.csv
│   ├── rfm_segmented.csv
│   ├── rfm_scaler.joblib
│   └── kmeans_model.joblib
├── notebooks/
│   ├── 01_cleaning.ipynb
│   ├── 02_eda.ipynb
│   └── 03_rfm_clustering.ipynb
├── src/
│   ├── cleaning.py
│   ├── rfm.py
│   └── clustering.py
├── streamlit_app/
│   └── app.py
├── requirements.txt
└── README.md
```

---

## Setup & Run

```bash
pip install -r requirements.txt
```

Artefacts under `data/` are already generated. To re-run the notebooks:

```bash
jupyter notebook notebooks/
```

Launch the dashboard:

```bash
cd streamlit_app
streamlit run app.py
```

The dashboard includes a light / dark mode toggle in the sidebar.

---

## Analysis Overview

**Data preparation**  
Temporal features, isolation of cancellations / anomalies / guest checkouts, line-level revenue and UK flag.

**Exploratory analysis**  
Monthly and seasonal patterns, hourly and weekday traffic, AOV, product Pareto, geography, returns, guest vs registered behaviour.

**RFM scoring**  
Recency, Frequency and Monetary scored into five quantiles and mapped to business segments (Champions, Loyal, At-Risk, New Inflow, …).

**Clustering**  
Log-transform + standardisation, K-Means (k = 4 via elbow & silhouette), hierarchical and DBSCAN comparison, 3-D RFM view.

**Dashboard**  
KPI cards, interactive Plotly charts, RFM explorer, segment profiles, geography, products, cancellations, time patterns, and marketing recommendations. Light and dark themes.

---

**Data source:** UCI Machine Learning Repository – Online Retail  
**Period:** 1 December 2010 – 9 December 2011
