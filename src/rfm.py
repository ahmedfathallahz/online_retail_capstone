"""
RFM scoring and business segmentation.

Recency   = days since last purchase
Frequency = number of unique invoices
Monetary  = total line revenue

Scores are 1–5 quantile ranks. For Recency the scale is reversed
(lower days → higher score). The resulting RFM string (e.g. "555")
and the business labels are what the marketing team actually uses.
"""

from __future__ import annotations

import logging
from datetime import timedelta

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_rfm(
    transactions: pd.DataFrame,
    analysis_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Aggregate transaction lines up to customer level.

    Extra columns kept for later profiling:
    - First / Last purchase dates
    - Tenure in days
    - Average order value
    - Total quantity purchased
    """
    if analysis_date is None:
        analysis_date = transactions["InvoiceDate"].max() + timedelta(days=1)

    rfm = (
        transactions.groupby("CustomerID")
        .agg(
            Recency=("InvoiceDate", lambda x: (analysis_date - x.max()).days),
            Frequency=("InvoiceNo", "nunique"),
            Monetary=("LineRevenue", "sum"),
            FirstPurchase=("InvoiceDate", "min"),
            LastPurchase=("InvoiceDate", "max"),
            TotalQuantity=("Quantity", "sum"),
            LineCount=("InvoiceNo", "count"),
        )
        .reset_index()
    )

    rfm["AvgOrderValue"] = rfm["Monetary"] / rfm["Frequency"]
    rfm["TenureDays"] = (rfm["LastPurchase"] - rfm["FirstPurchase"]).dt.days
    rfm["AvgBasketSize"] = rfm["TotalQuantity"] / rfm["Frequency"]

    logger.info("RFM table: %s customers", len(rfm))
    return rfm


def score_rfm(rfm: pd.DataFrame, n_quantiles: int = 5) -> pd.DataFrame:
    """
    Assign 1–5 scores with pd.qcut.
    Recency is inverted so that 5 always means “best”.
    """
    rfm = rfm.copy()

    rfm["R_Score"] = pd.qcut(
        rfm["Recency"],
        q=n_quantiles,
        labels=[5, 4, 3, 2, 1],
        duplicates="drop",
    ).astype(int)

    # rank(method="first") avoids ties that break qcut on discrete Frequency
    rfm["F_Score"] = pd.qcut(
        rfm["Frequency"].rank(method="first"),
        q=n_quantiles,
        labels=[1, 2, 3, 4, 5],
        duplicates="drop",
    ).astype(int)

    rfm["M_Score"] = pd.qcut(
        rfm["Monetary"].rank(method="first"),
        q=n_quantiles,
        labels=[1, 2, 3, 4, 5],
        duplicates="drop",
    ).astype(int)

    rfm["RFM_Score"] = (
        rfm["R_Score"].astype(str)
        + rfm["F_Score"].astype(str)
        + rfm["M_Score"].astype(str)
    )
    rfm["RFM_Sum"] = rfm["R_Score"] + rfm["F_Score"] + rfm["M_Score"]

    return rfm


def segment_customers(rfm: pd.DataFrame) -> pd.DataFrame:
    """
    Map RFM scores onto actionable business segments.

    The rules follow the classic matrix used by most retail CRM teams,
    adjusted slightly so the largest value pockets are cleanly separated.
    """
    rfm = rfm.copy()

    def _label(row):
        r, f, m = row["R_Score"], row["F_Score"], row["M_Score"]

        if r == 5 and f == 5 and m == 5:
            return "Champions"
        if r >= 4 and f >= 4 and m >= 4:
            return "Loyal Customers"
        if r <= 2 and f >= 4 and m >= 4:
            return "At-Risk / High-Value Churn"
        if r >= 4 and f <= 2 and m <= 2:
            return "New Inflow"
        if r >= 4 and f >= 3 and m >= 3:
            return "Potential Loyalists"
        if r <= 2 and f <= 2 and m <= 2:
            return "Lost / Hibernating"
        if r <= 2 and (f >= 3 or m >= 3):
            return "Need Attention"
        if r >= 3 and f <= 2 and m >= 3:
            return "Promising"
        return "Others"

    rfm["Segment"] = rfm.apply(_label, axis=1)
    return rfm


def full_rfm_pipeline(transactions: pd.DataFrame) -> pd.DataFrame:
    """Convenience wrapper used by the notebooks and the Streamlit app."""
    rfm = compute_rfm(transactions)
    rfm = score_rfm(rfm)
    rfm = segment_customers(rfm)
    return rfm
