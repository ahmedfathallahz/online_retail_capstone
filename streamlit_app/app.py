"""
Online Retail Customer Analytics & Segmentation
Interactive Streamlit dashboard with a fixed dark theme.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

st.set_page_config(
    page_title="Retail Customer Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────
# Fixed dark theme
# ──────────────────────────────────────────────────────────────
is_dark = True

BG          = "#0f172a"
CARD_BG     = "#1e293b"
CARD_BORDER = "#334155"
TEXT        = "#f1f5f9"
MUTED       = "#94a3b8"
ACCENT      = "#60a5fa"
SIDEBAR_BG  = "linear-gradient(180deg, #020617 0%, #0f172a 100%)"
PLOTLY_TPL  = "plotly_dark"
KPI_SHADOW  = "0 4px 16px rgba(0,0,0,0.35)"

# ──────────────────────────────────────────────────────────────
# CSS
# ──────────────────────────────────────────────────────────────
st.html(
    f"""
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">
    <style>
    :root {{ color-scheme: dark; }}
    .stApp, [data-testid="stAppViewContainer"] {{
        background-color: {BG} !important;
        color: {TEXT} !important;
        --text-color: {TEXT};
        --background-color: {BG};
        --secondary-background-color: {CARD_BG};
        --primary-color: {ACCENT};
    }}
    .main .block-container {{
        padding-top: 1.4rem;
        padding-bottom: 2.5rem;
        max-width: 1280px;
    }}
    .main h1, .main h2, .main h3, .main p,
    .main .stMarkdown, .main [data-testid="stMetricLabel"],
    .main [data-testid="stMetricValue"],
    .main [data-testid="stWidgetLabel"] {{ color: {TEXT} !important; }}
    .main [data-baseweb="select"] *,
    .main [data-baseweb="input"] *,
    .main [data-baseweb="textarea"] * {{ color: {TEXT} !important; }}
    .main [data-baseweb="select"] > div,
    .main [data-baseweb="input"] > div,
    .main [data-baseweb="textarea"] > div {{
        background-color: {CARD_BG} !important;
        border-color: {CARD_BORDER} !important;
    }}
    .main [data-testid="stDataFrame"], .main [data-testid="stTable"] {{
        color: {TEXT} !important;
    }}
    .main [data-testid="stCaptionContainer"],
    .main [data-testid="stMarkdownContainer"],
    .main [data-testid="stText"],
    .main [data-testid="stAlert"] {{
        color: {TEXT} !important;
    }}
    .main [data-testid="stPlotlyChart"] {{
        color: {TEXT} !important;
    }}

    .kpi-card {{
        background: {CARD_BG};
        border-radius: 14px;
        padding: 1.3rem 1.4rem;
        box-shadow: {KPI_SHADOW};
        border: 1px solid {CARD_BORDER};
        text-align: center;
        height: 100%;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }}
    .kpi-card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.12);
    }}
    .kpi-card .icon {{
        font-size: 1.55rem;
        color: {ACCENT};
        margin-bottom: 0.45rem;
    }}
    .kpi-card .value {{
        font-size: 1.7rem;
        font-weight: 700;
        color: {TEXT};
        line-height: 1.2;
    }}
    .kpi-card .label {{
        font-size: 0.8rem;
        color: {MUTED};
        margin-top: 0.3rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}

    .section-header {{
        display: flex;
        align-items: center;
        gap: 0.55rem;
        font-size: 1.28rem;
        font-weight: 650;
        color: {TEXT};
        margin: 1.6rem 0 0.7rem 0;
    }}
    .section-header i {{ color: {ACCENT}; }}

    .rec-card {{
        background: {CARD_BG};
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 0.9rem;
        box-shadow: {KPI_SHADOW};
        border: 1px solid {CARD_BORDER};
    }}

    [data-testid="stSidebar"] {{
        background: {SIDEBAR_BG} !important;
        border-right: 1px solid {CARD_BORDER};
    }}
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {{
        color: #e2e8f0 !important;
    }}
    /* Icon + label navigation rows */
    [data-testid="stSidebar"] [class*="st-key-nav_"] button {{
        width: 100%;
        min-height: 48px;
        justify-content: flex-start !important;
        text-align: left !important;
        border-radius: 11px !important;
        border: 1px solid rgba(148,163,184,0.24) !important;
        background: rgba(255,255,255,0.055) !important;
        color: #e2e8f0 !important;
        padding: 0.55rem 0.85rem !important;
        margin: 0.12rem 0 !important;
        transition: all 0.16s ease;
    }}
    [data-testid="stSidebar"] [class*="st-key-nav_"] button:hover {{
        border-color: {ACCENT} !important;
        background: rgba(96,165,250,0.16) !important;
        transform: translateX(2px);
    }}
    [data-testid="stSidebar"] [class*="st-key-nav_"] button[kind="primary"] {{
        background: {ACCENT} !important;
        border-color: {ACCENT} !important;
        color: #ffffff !important;
        box-shadow: 0 4px 12px rgba(59,130,246,0.22);
    }}
    [data-testid="stSidebar"] [class*="st-key-nav_"] button[kind="primary"] * {{
        color: #ffffff !important;
        fill: #ffffff !important;
    }}
    [data-testid="stSidebar"] [class*="st-key-nav_"] button p {{
        color: inherit !important;
        font-weight: 600;
        margin: 0 !important;
    }}
    [data-testid="stSidebar"] .st-key-theme_toggle button {{
        border: 1px solid #64748b !important;
        color: #f8fafc !important;
        background: rgba(255,255,255,0.08) !important;
    }}
    [data-testid="stSidebar"] .st-key-theme_toggle button:hover {{
        border-color: {ACCENT} !important;
        background: rgba(96,165,250,0.16) !important;
    }}

    [data-testid="stHeader"] {{ background: transparent; }}
    [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"] {{
        color: {TEXT};
    }}
    [data-baseweb="popover"], [data-baseweb="menu"] {{
        background-color: {CARD_BG} !important;
        color: {TEXT} !important;
    }}
    [data-baseweb="popover"] *, [data-baseweb="menu"] * {{
        color: {TEXT} !important;
    }}
    .stButton > button {{ color: {TEXT}; }}

    .footer {{
        text-align: center;
        color: {MUTED};
        font-size: 0.78rem;
        margin-top: 2.5rem;
        padding-top: 1rem;
        border-top: 1px solid {CARD_BORDER};
    }}

    #MainMenu {{ visibility: hidden; }}
    footer {{ visibility: hidden; }}
    </style>
    """,
)


# ──────────────────────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading data…")
def load_data():
    rfm_path = ROOT / "data" / "rfm_segmented.csv"
    reg_path = ROOT / "data" / "registered_clean.csv"
    guest_path = ROOT / "data" / "guests.csv"
    cancel_path = ROOT / "data" / "cancellations.csv"

    if not rfm_path.exists() or not reg_path.exists():
        st.error("Required data files not found. Run the notebooks first.")
        st.stop()

    rfm = pd.read_csv(rfm_path)
    registered = pd.read_csv(reg_path, parse_dates=["InvoiceDate"])
    guests = pd.read_csv(guest_path, parse_dates=["InvoiceDate"]) if guest_path.exists() else None
    cancels = pd.read_csv(cancel_path, parse_dates=["InvoiceDate"]) if cancel_path.exists() else None
    return rfm, registered, guests, cancels


rfm, registered, guests, cancels = load_data()


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────
def section(icon: str, title: str):
    st.markdown(
        f'<div class="section-header"><i class="fa-solid {icon}"></i> {title}</div>',
        unsafe_allow_html=True,
    )


def kpi(icon: str, value: str, label: str) -> str:
    return f"""
    <div class="kpi-card">
        <div class="icon"><i class="fa-solid {icon}"></i></div>
        <div class="value">{value}</div>
        <div class="label">{label}</div>
    </div>
    """


def apply_layout(fig, height=400):
    # Explicitly set every Plotly text layer. Relying on the template alone
    # leaves some axis/colorbar labels too pale in the light theme.
    axis_text = "#e2e8f0"
    axis_muted = "#cbd5e1"
    grid = "#334155"

    fig.update_layout(
        template=PLOTLY_TPL,
        height=height,
        margin=dict(l=16, r=16, t=36, b=16),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family="Arial, sans-serif",
            size=13,
            color=TEXT,
        ),
        title_font=dict(color=TEXT),
        legend=dict(font=dict(color=axis_text, size=12)),
        coloraxis_colorbar=dict(
            tickfont=dict(color=axis_text, size=11),
            title_font=dict(color=axis_text, size=12),
            outlinecolor=grid,
        ),
    )
    fig.update_xaxes(
        title_font=dict(color=axis_text, size=12),
        tickfont=dict(color=axis_muted, size=11),
        gridcolor=grid,
        zerolinecolor=grid,
        linecolor=grid,
    )
    fig.update_yaxes(
        title_font=dict(color=axis_text, size=12),
        tickfont=dict(color=axis_muted, size=11),
        gridcolor=grid,
        zerolinecolor=grid,
        linecolor=grid,
    )
    # Heatmaps and some Plotly Express charts use annotations for labels.
    fig.update_annotations(font=dict(color=axis_text))
    return fig


# ──────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center; padding: 0.6rem 0 1rem 0;">
            <i class="fa-solid fa-store" style="font-size:2.1rem; color:#60a5fa;"></i>
            <h2 style="margin:0.4rem 0 0 0; color:white; font-size:1.25rem;">Retail Analytics</h2>
            <p style="margin:0; font-size:0.72rem; color:#94a3b8;">Customer Segmentation</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Each navigation item is a full-width button with a Material icon.
    nav_items = [
        ("Overview", "dashboard"),
        ("RFM Explorer", "tune"),
        ("Segment Profiles", "groups"),
        ("Geographic Insights", "public"),
        ("Product Insights", "inventory_2"),
        ("Cancellations & Returns", "assignment_return"),
        ("Time Patterns", "schedule"),
        ("Business Recommendations", "lightbulb"),
    ]
    if "page" not in st.session_state:
        st.session_state.page = "Overview"

    def set_page(selected_page):
        st.session_state.page = selected_page

    for nav_label, nav_icon in nav_items:
        nav_key = "nav_" + nav_label.lower().replace(" ", "_").replace("&", "and")
        st.button(
            nav_label,
            icon=f":material/{nav_icon}:",
            key=nav_key,
            type="primary" if st.session_state.page == nav_label else "secondary",
            use_container_width=True,
            on_click=set_page,
            args=(nav_label,),
        )

    page = st.session_state.page

    st.markdown("---")


    st.markdown(
        """
        <div style="font-size:0.72rem; color:#64748b; padding:0.8rem 0.4rem 0;">
        UCI Online Retail<br>
        Dec 2010 – Dec 2011<br>
        RFM + K-Means
        </div>
        """,
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────────────────────
# PAGE: Overview
# ──────────────────────────────────────────────────────────────
if page == "Overview":
    st.markdown(
        f"""
        <h1 style="margin-bottom:0.15rem; color:{TEXT};">
            <i class="fa-solid fa-chart-line" style="color:{ACCENT};"></i>
            Customer Analytics Overview
        </h1>
        <p style="color:{MUTED}; margin-top:0; margin-bottom:1.2rem;">
            Revenue, customers and engagement for a UK online gift retailer.
        </p>
        """,
        unsafe_allow_html=True,
    )

    total_rev = registered["LineRevenue"].sum()
    n_cust = rfm["CustomerID"].nunique()
    n_orders = registered["InvoiceNo"].nunique()
    aov = total_rev / n_orders
    guest_rev = guests["LineRevenue"].sum() if guests is not None else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.markdown(kpi("fa-coins", f"£{total_rev/1e6:.2f}M", "Registered Revenue"), unsafe_allow_html=True)
    c2.markdown(kpi("fa-users", f"{n_cust:,}", "Customers"), unsafe_allow_html=True)
    c3.markdown(kpi("fa-receipt", f"{n_orders:,}", "Orders"), unsafe_allow_html=True)
    c4.markdown(kpi("fa-basket-shopping", f"£{aov:.0f}", "Avg Order Value"), unsafe_allow_html=True)
    c5.markdown(kpi("fa-user-secret", f"£{guest_rev/1e6:.2f}M", "Guest Revenue"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    section("fa-calendar-days", "Monthly Revenue Trend")
    monthly = (
        registered.groupby(registered["InvoiceDate"].dt.to_period("M"))
        .agg(Revenue=("LineRevenue", "sum"), Orders=("InvoiceNo", "nunique"))
        .reset_index()
    )
    monthly["Month"] = monthly["InvoiceDate"].astype(str)

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=monthly["Month"], y=monthly["Revenue"], name="Revenue",
            fill="tozeroy", line=dict(color=ACCENT),
            fillcolor="rgba(37,99,235,0.18)",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=monthly["Month"], y=monthly["Orders"], name="Orders",
            line=dict(color="#2563eb", dash="dot"),
        ),
        secondary_y=True,
    )
    fig = apply_layout(fig, 400)
    fig.update_layout(legend=dict(orientation="h", y=1.12))
    fig.update_yaxes(title_text="Revenue", tickprefix="£", secondary_y=False, gridcolor=CARD_BORDER)
    fig.update_yaxes(title_text="Orders", secondary_y=True, gridcolor=CARD_BORDER)
    st.plotly_chart(fig, use_container_width=True)

    col_l, col_r = st.columns(2)
    with col_l:
        section("fa-layer-group", "Customer Segment Mix")
        seg_counts = rfm["Segment"].value_counts().reset_index()
        seg_counts.columns = ["Segment", "Customers"]
        fig2 = px.pie(
            seg_counts, names="Segment", values="Customers", hole=0.45,
            color_discrete_sequence=["#0b3c78", "#145da0", "#1976b9", "#268bd2", "#5aa9e6", "#8ecae6", "#bde0fe"],
        )
        fig2 = apply_layout(fig2, 380)
        fig2.update_layout(legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig2, use_container_width=True)

    with col_r:
        section("fa-globe", "Revenue by Country (Top 8)")
        geo = (
            registered.groupby("Country")["LineRevenue"].sum()
            .sort_values(ascending=False).head(8).reset_index()
        )
        fig3 = px.bar(
            geo, x="LineRevenue", y="Country", orientation="h",
            color="LineRevenue", color_continuous_scale="Blues",
        )
        fig3 = apply_layout(fig3, 380)
        fig3.update_layout(showlegend=False, coloraxis_showscale=False,
                           xaxis_title="Revenue", yaxis_title="")
        fig3.update_yaxes(categoryorder="total ascending")
        st.plotly_chart(fig3, use_container_width=True)

    section("fa-table-cells", "RFM Score Heatmap")
    heat = pd.crosstab(rfm["R_Score"], rfm["F_Score"])
    fig_h = px.imshow(
        heat, text_auto=True, color_continuous_scale="Blues",
        labels=dict(x="Frequency Score", y="Recency Score", color="Customers"),
    )
    fig_h = apply_layout(fig_h, 340)
    st.plotly_chart(fig_h, use_container_width=True)


# ──────────────────────────────────────────────────────────────
# PAGE: RFM Explorer
# ──────────────────────────────────────────────────────────────
elif page == "RFM Explorer":
    st.markdown(
        f"""
        <h1 style="margin-bottom:0.15rem; color:{TEXT};">
            <i class="fa-solid fa-sliders" style="color:{ACCENT};"></i>
            RFM Explorer
        </h1>
        <p style="color:{MUTED};">
            Filter by Recency, Frequency and Monetary scores. Higher = better (R=5 is most recent).
        </p>
        """,
        unsafe_allow_html=True,
    )

    f1, f2, f3, f4 = st.columns(4)
    with f1:
        r_sel = st.multiselect("Recency", options=[1, 2, 3, 4, 5], default=[1, 2, 3, 4, 5])
    with f2:
        f_sel = st.multiselect("Frequency", options=[1, 2, 3, 4, 5], default=[1, 2, 3, 4, 5])
    with f3:
        m_sel = st.multiselect("Monetary", options=[1, 2, 3, 4, 5], default=[1, 2, 3, 4, 5])
    with f4:
        seg_sel = st.multiselect(
            "Segment",
            options=sorted(rfm["Segment"].unique()),
            default=sorted(rfm["Segment"].unique()),
        )

    filtered = rfm[
        rfm["R_Score"].isin(r_sel)
        & rfm["F_Score"].isin(f_sel)
        & rfm["M_Score"].isin(m_sel)
        & rfm["Segment"].isin(seg_sel)
    ]

    st.markdown(
        f"**{len(filtered):,}** customers · £{filtered['Monetary'].sum():,.0f} revenue"
    )

    section("fa-chart-scatter", "Recency × Frequency (size = Monetary)")
    sample = filtered.sample(min(3000, len(filtered)), random_state=42) if len(filtered) else filtered
    if len(sample):
        fig = px.scatter(
            sample, x="Recency", y="Frequency", size="Monetary", color="Segment",
            hover_data=["CustomerID", "RFM_Score", "Monetary"],
            log_y=True, size_max=25, opacity=0.65,
            color_discrete_sequence=["#0b3c78", "#145da0", "#1976b9", "#268bd2", "#5aa9e6", "#8ecae6", "#bde0fe"],
        )
        fig = apply_layout(fig, 480)
        fig.update_layout(
            xaxis_title="Recency (days)", yaxis_title="Frequency (invoices)",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No customers match the selected filters.")

    section("fa-chart-column", "Score distributions")
    c1, c2, c3 = st.columns(3)
    for col, score_col, title, color in [
        (c1, "R_Score", "Recency", "#3b82f6"),
        (c2, "F_Score", "Frequency", "#1976b9"),
        (c3, "M_Score", "Monetary", "#5aa9e6"),
    ]:
        with col:
            counts = filtered[score_col].value_counts().sort_index()
            fig = px.bar(
                x=counts.index.astype(str), y=counts.values,
                labels={"x": "Score", "y": "Customers"}, title=title,
            )
            fig.update_traces(marker_color=color)
            fig = apply_layout(fig, 280)
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    section("fa-table", "Customer table")
    show_cols = [
        "CustomerID", "Recency", "Frequency", "Monetary",
        "R_Score", "F_Score", "M_Score", "RFM_Score", "Segment", "Cluster_KMeans",
    ]
    available = [c for c in show_cols if c in filtered.columns]
    st.dataframe(
        filtered[available].sort_values("Monetary", ascending=False).head(500),
        use_container_width=True, height=350,
    )


# ──────────────────────────────────────────────────────────────
# PAGE: Segment Profiles
# ──────────────────────────────────────────────────────────────
elif page == "Segment Profiles":
    st.markdown(
        f"""
        <h1 style="margin-bottom:0.15rem; color:{TEXT};">
            <i class="fa-solid fa-users-viewfinder" style="color:{ACCENT};"></i>
            Segment Profiles
        </h1>
        <p style="color:{MUTED};">Business segments from RFM scores and K-Means clusters.</p>
        """,
        unsafe_allow_html=True,
    )

    profile = (
        rfm.groupby("Segment")
        .agg(
            Customers=("CustomerID", "count"),
            Avg_Recency=("Recency", "mean"),
            Avg_Frequency=("Frequency", "mean"),
            Avg_Monetary=("Monetary", "mean"),
            Total_Revenue=("Monetary", "sum"),
        )
        .round(1)
        .sort_values("Total_Revenue", ascending=False)
        .reset_index()
    )
    profile["Revenue_Share_%"] = (
        profile["Total_Revenue"] / profile["Total_Revenue"].sum() * 100
    ).round(1)

    section("fa-table-list", "Segment summary")
    st.dataframe(profile, use_container_width=True, hide_index=True)

    section("fa-chart-radar", "Average RFM by segment")
    fig = go.Figure()
    for _, row in profile.iterrows():
        fig.add_trace(
            go.Scatterpolar(
                r=[row["Avg_Recency"], row["Avg_Frequency"], row["Avg_Monetary"] / 100],
                theta=["Recency (days)", "Frequency", "Monetary (/100)"],
                fill="toself", name=row["Segment"],
            )
        )
    fig = apply_layout(fig, 500)
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, gridcolor=CARD_BORDER)),
        showlegend=True, legend=dict(orientation="h", y=-0.2),
    )
    st.plotly_chart(fig, use_container_width=True)

    if "Cluster_KMeans" in rfm.columns:
        section("fa-diagram-project", "K-Means clusters")
        clus = (
            rfm.groupby("Cluster_KMeans")
            .agg(
                N=("CustomerID", "count"),
                AvgR=("Recency", "mean"),
                AvgF=("Frequency", "mean"),
                AvgM=("Monetary", "mean"),
                TotalRev=("Monetary", "sum"),
            )
            .round(1).reset_index()
        )
        st.dataframe(clus, use_container_width=True, hide_index=True)

        fig2 = px.scatter_3d(
            rfm.sample(min(2000, len(rfm)), random_state=42),
            x="Recency", y="Frequency", z="Monetary",
            color="Cluster_KMeans", opacity=0.75, log_y=True, log_z=True,
            color_discrete_sequence=["#0b3c78", "#145da0", "#1976b9", "#268bd2", "#5aa9e6", "#8ecae6"],
            title="K-Means in 3-D RFM space",
        )
        fig2 = apply_layout(fig2, 540)
        st.plotly_chart(fig2, use_container_width=True)

        section("fa-table-cells", "Segment × Cluster")
        st.dataframe(pd.crosstab(rfm["Segment"], rfm["Cluster_KMeans"]), use_container_width=True)


# ──────────────────────────────────────────────────────────────
# PAGE: Geographic Insights
# ──────────────────────────────────────────────────────────────
elif page == "Geographic Insights":
    st.markdown(
        f"""
        <h1 style="margin-bottom:0.15rem; color:{TEXT};">
            <i class="fa-solid fa-earth-europe" style="color:{ACCENT};"></i>
            Geographic Insights
        </h1>
        """,
        unsafe_allow_html=True,
    )

    geo = (
        registered.groupby("Country")
        .agg(
            Revenue=("LineRevenue", "sum"),
            Customers=("CustomerID", "nunique"),
            Orders=("InvoiceNo", "nunique"),
            Items=("Quantity", "sum"),
        )
        .sort_values("Revenue", ascending=False).reset_index()
    )
    geo["AOV"] = geo["Revenue"] / geo["Orders"]
    geo["RevenueShare"] = geo["Revenue"] / geo["Revenue"].sum() * 100
    geo["RevPerCustomer"] = geo["Revenue"] / geo["Customers"]

    section("fa-trophy", "Top markets by revenue")
    fig = px.bar(
        geo.head(15), x="Country", y="Revenue",
        text=geo.head(15)["RevenueShare"].round(1).astype(str) + "%",
        color="Revenue", color_continuous_scale="Blues",
    )
    fig = apply_layout(fig, 440)
    fig.update_layout(
        xaxis_tickangle=-35, yaxis_tickprefix="£",
        coloraxis_showscale=False, margin=dict(l=20, r=20, t=20, b=80),
    )
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        section("fa-chart-simple", "Revenue per customer (non-UK)")
        non_uk = geo[geo["Country"] != "United Kingdom"].head(12)
        fig2 = px.bar(
            non_uk, x="RevPerCustomer", y="Country", orientation="h",
            color="RevPerCustomer", color_continuous_scale="Blues",
        )
        fig2 = apply_layout(fig2, 400)
        fig2.update_layout(
            coloraxis_showscale=False,
            yaxis={"categoryorder": "total ascending"},
            xaxis_tickprefix="£",
        )
        st.plotly_chart(fig2, use_container_width=True)
    with c2:
        section("fa-basket-shopping", "AOV by country")
        fig3 = px.bar(
            geo.head(12), x="AOV", y="Country", orientation="h",
            color="AOV", color_continuous_scale="Blues",
        )
        fig3 = apply_layout(fig3, 400)
        fig3.update_layout(
            coloraxis_showscale=False,
            yaxis={"categoryorder": "total ascending"},
            xaxis_tickprefix="£",
        )
        st.plotly_chart(fig3, use_container_width=True)

    section("fa-table", "Full country table")
    st.dataframe(
        geo.style.format({
            "Revenue": "£{:,.0f}", "AOV": "£{:,.1f}",
            "RevenueShare": "{:.2f}%", "RevPerCustomer": "£{:,.0f}",
        }),
        use_container_width=True, height=420,
    )


# ──────────────────────────────────────────────────────────────
# PAGE: Product Insights
# ──────────────────────────────────────────────────────────────
elif page == "Product Insights":
    st.markdown(
        f"""
        <h1 style="margin-bottom:0.15rem; color:{TEXT};">
            <i class="fa-solid fa-box-open" style="color:{ACCENT};"></i>
            Product Insights
        </h1>
        """,
        unsafe_allow_html=True,
    )

    prod = (
        registered.groupby(["StockCode", "Description"])
        .agg(
            Revenue=("LineRevenue", "sum"),
            Quantity=("Quantity", "sum"),
            Orders=("InvoiceNo", "nunique"),
        )
        .sort_values("Revenue", ascending=False).reset_index()
    )
    prod["CumPct"] = prod["Revenue"].cumsum() / prod["Revenue"].sum() * 100

    section("fa-star", "Top 20 products by revenue")
    fig = px.bar(
        prod.head(20), x="Revenue", y="Description", orientation="h",
        color="Revenue", color_continuous_scale="Blues",
    )
    fig = apply_layout(fig, 580)
    fig.update_layout(
        yaxis={"categoryorder": "total ascending"},
        coloraxis_showscale=False, xaxis_tickprefix="£",
    )
    st.plotly_chart(fig, use_container_width=True)

    section("fa-chart-line", "Pareto curve")
    fig2 = px.line(
        prod.head(400), x=prod.head(400).index + 1, y="CumPct",
        labels={"x": "Product rank", "CumPct": "Cumulative revenue %"},
    )
    fig2.add_hline(y=80, line_dash="dash", line_color="#2563eb", annotation_text="80%")
    fig2 = apply_layout(fig2, 380)
    st.plotly_chart(fig2, use_container_width=True)

    n80 = (prod["CumPct"] <= 80).sum()
    st.info(f"**{n80} products** ({n80/len(prod)*100:.1f}% of catalogue) generate 80% of revenue.")

    section("fa-table", "Top 50 products")
    st.dataframe(
        prod.head(50)[["StockCode", "Description", "Revenue", "Quantity", "Orders", "CumPct"]]
        .style.format({"Revenue": "£{:,.0f}", "CumPct": "{:.1f}%"}),
        use_container_width=True, height=380,
    )


# ──────────────────────────────────────────────────────────────
# PAGE: Cancellations & Returns
# ──────────────────────────────────────────────────────────────
elif page == "Cancellations & Returns":
    st.markdown(
        f"""
        <h1 style="margin-bottom:0.15rem; color:{TEXT};">
            <i class="fa-solid fa-rotate-left" style="color:{ACCENT};"></i>
            Cancellations & Returns
        </h1>
        <p style="color:{MUTED};">
            Invoice numbers starting with “C” carry negative quantities.
        </p>
        """,
        unsafe_allow_html=True,
    )

    if cancels is None or len(cancels) == 0:
        st.warning("Cancellation data not available.")
    else:
        cancel_val = cancels["LineRevenue"].abs().sum()
        c1, c2, c3 = st.columns(3)
        c1.markdown(kpi("fa-receipt", f"{len(cancels):,}", "Cancel Lines"), unsafe_allow_html=True)
        c2.markdown(kpi("fa-coins", f"£{cancel_val:,.0f}", "Cancel Value"), unsafe_allow_html=True)
        c3.markdown(
            kpi("fa-users", f"{cancels['CustomerID'].nunique():,}", "Customers Involved"),
            unsafe_allow_html=True,
        )

        section("fa-box", "Most returned products")
        top_ret = (
            cancels.groupby(["StockCode", "Description"])
            .agg(
                ReturnQty=("Quantity", lambda x: x.abs().sum()),
                ReturnValue=("LineRevenue", lambda x: x.abs().sum()),
            )
            .sort_values("ReturnValue", ascending=False).head(20).reset_index()
        )
        fig = px.bar(
            top_ret, x="ReturnValue", y="Description", orientation="h",
            color="ReturnValue", color_continuous_scale="Blues",
        )
        fig = apply_layout(fig, 560)
        fig.update_layout(
            yaxis={"categoryorder": "total ascending"},
            coloraxis_showscale=False, xaxis_tickprefix="£",
        )
        st.plotly_chart(fig, use_container_width=True)

        section("fa-calendar", "Cancellations by month")
        cancels = cancels.copy()
        cancels["Month"] = cancels["InvoiceDate"].dt.to_period("M").astype(str)
        monthly_c = cancels.groupby("Month").agg(
            Value=("LineRevenue", lambda x: x.abs().sum()),
        ).reset_index()
        fig2 = px.bar(monthly_c, x="Month", y="Value", labels={"Value": "Cancel value"})
        fig2.update_traces(marker_color="#2563eb")
        fig2 = apply_layout(fig2, 360)
        fig2.update_layout(yaxis_tickprefix="£")
        st.plotly_chart(fig2, use_container_width=True)


# ──────────────────────────────────────────────────────────────
# PAGE: Time Patterns
# ──────────────────────────────────────────────────────────────
elif page == "Time Patterns":
    st.markdown(
        f"""
        <h1 style="margin-bottom:0.15rem; color:{TEXT};">
            <i class="fa-solid fa-clock" style="color:{ACCENT};"></i>
            Time Patterns
        </h1>
        <p style="color:{MUTED};">
            Hour-of-day, weekday and seasonal behaviour.
        </p>
        """,
        unsafe_allow_html=True,
    )

    section("fa-clock", "Orders by hour")
    hourly = (
        registered.groupby("Hour")
        .agg(Orders=("InvoiceNo", "nunique"), Revenue=("LineRevenue", "sum"))
        .reset_index()
    )
    fig = px.bar(hourly, x="Hour", y="Orders")
    fig.update_traces(marker_color="#1976b9")
    fig = apply_layout(fig, 380)
    st.plotly_chart(fig, use_container_width=True)

    section("fa-calendar-week", "Revenue by weekday")
    dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    dow = (
        registered.groupby("DayOfWeek")
        .agg(Revenue=("LineRevenue", "sum"), Orders=("InvoiceNo", "nunique"))
        .reindex(dow_order).reset_index()
    )
    fig2 = px.bar(dow, x="DayOfWeek", y="Revenue")
    fig2.update_traces(marker_color="#5aa9e6")
    fig2 = apply_layout(fig2, 380)
    fig2.update_layout(yaxis_tickprefix="£")
    st.plotly_chart(fig2, use_container_width=True)

    section("fa-calendar-days", "Hour × Weekday heatmap")
    registered = registered.copy()
    registered["Dow"] = registered["InvoiceDate"].dt.day_name()
    heat = (
        registered.groupby(["Dow", "Hour"])["InvoiceNo"].nunique()
        .unstack(fill_value=0)
        .reindex(dow_order)
    )
    fig3 = px.imshow(
        heat, aspect="auto", color_continuous_scale="Blues",
        labels=dict(x="Hour", y="Weekday", color="Orders"),
    )
    fig3 = apply_layout(fig3, 400)
    st.plotly_chart(fig3, use_container_width=True)


# ──────────────────────────────────────────────────────────────
# PAGE: Business Recommendations
# ──────────────────────────────────────────────────────────────
elif page == "Business Recommendations":
    st.markdown(
        f"""
        <h1 style="margin-bottom:0.15rem; color:{TEXT};">
            <i class="fa-solid fa-lightbulb" style="color:{ACCENT};"></i>
            Business Recommendations
        </h1>
        <p style="color:{MUTED};">
            Concrete actions mapped from the RFM segments.
        </p>
        """,
        unsafe_allow_html=True,
    )

    seg_counts = rfm["Segment"].value_counts()
    seg_rev = rfm.groupby("Segment")["Monetary"].sum()

    recommendations = [
        {
            "icon": "fa-crown",
            "title": "Champions (R5-F5-M5)",
            "color": "#f59e0b",
            "n": int(seg_counts.get("Champions", 0)),
            "rev": float(seg_rev.get("Champions", 0)),
            "strategy": (
                "Most profitable and engaged buyers. Reward with loyalty perks, "
                "exclusive early access, and referral programs. Avoid heavy discounting."
            ),
        },
        {
            "icon": "fa-heart",
            "title": "Loyal Customers",
            "color": "#10b981",
            "n": int(seg_counts.get("Loyal Customers", 0)),
            "rev": float(seg_rev.get("Loyal Customers", 0)),
            "strategy": (
                "Consistent spenders. Maximise lifetime value through cross-sell "
                "and personalised product recommendations."
            ),
        },
        {
            "icon": "fa-triangle-exclamation",
            "title": "At-Risk / High-Value Churn",
            "color": "#ef4444",
            "n": int(seg_counts.get("At-Risk / High-Value Churn", 0)),
            "rev": float(seg_rev.get("At-Risk / High-Value Churn", 0)),
            "strategy": (
                "Heavy historic spenders who went quiet. Deploy immediate win-back "
                "campaigns with strong price triggers."
            ),
        },
        {
            "icon": "fa-user-plus",
            "title": "New Inflow",
            "color": "#3b82f6",
            "n": int(seg_counts.get("New Inflow", 0)),
            "rev": float(seg_rev.get("New Inflow", 0)),
            "strategy": (
                "Recent first-time buyers. Onboard with a welcome sequence designed "
                "to trigger a second purchase."
            ),
        },
        {
            "icon": "fa-clock-rotate-left",
            "title": "Lost / Hibernating",
            "color": "#64748b",
            "n": int(seg_counts.get("Lost / Hibernating", 0)),
            "rev": float(seg_rev.get("Lost / Hibernating", 0)),
            "strategy": (
                "Low engagement across all dimensions. Run low-cost reactivation tests; "
                "reduce pressure if no response."
            ),
        },
        {
            "icon": "fa-bell",
            "title": "Need Attention",
            "color": "#8b5cf6",
            "n": int(seg_counts.get("Need Attention", 0)),
            "rev": float(seg_rev.get("Need Attention", 0)),
            "strategy": (
                "Decent historic value but long silence. A short personalised "
                "re-engagement series is usually enough to bring a portion back."
            ),
        },
    ]

    for rec in recommendations:
        st.markdown(
            f"""
            <div class="rec-card" style="border-left:5px solid {rec['color']};">
                <div style="display:flex; align-items:center; gap:0.65rem; margin-bottom:0.35rem;">
                    <i class="fa-solid {rec['icon']}" style="color:{rec['color']}; font-size:1.25rem;"></i>
                    <strong style="font-size:1.05rem; color:{TEXT};">{rec['title']}</strong>
                </div>
                <div style="font-size:0.82rem; color:{MUTED}; margin-bottom:0.45rem;">
                    {rec['n']:,} customers · £{rec['rev']:,.0f} revenue
                </div>
                <p style="margin:0; color:{MUTED}; line-height:1.55;">{rec['strategy']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ──────────────────────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <div class="footer">
        Online Retail Customer Analytics &amp; Segmentation<br>
        UCI Machine Learning Repository · Streamlit + Plotly
    </div>
    """,
    unsafe_allow_html=True,
)
