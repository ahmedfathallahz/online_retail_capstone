"""
Data cleaning pipeline for the Online Retail dataset.

Every decision here is driven by how the business actually works:
- Guest checkouts still generate revenue, so we never drop them silently.
- Cancellations carry negative quantities; we isolate them so return rates
  can be measured later.
- Zero / negative prices are usually system adjustments or giveaways;
  they go into their own frame.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_raw(path: str | Path) -> pd.DataFrame:
    """Load the UCI Online Retail file (xlsx or csv)."""
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path, engine="openpyxl")
    else:
        df = pd.read_csv(path, parse_dates=["InvoiceDate"])
    logger.info("Raw load: %s rows × %s cols", *df.shape)
    return df


def structural_cleansing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Type casting + temporal features.

    Business reason for the date parts:
    - Month / Year → seasonal revenue patterns and Q4 surge detection
    - DayOfWeek / Hour → staffing and ad-spend timing
    """
    df = df.copy()

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df["Year"] = df["InvoiceDate"].dt.year
    df["Month"] = df["InvoiceDate"].dt.month
    df["DayOfWeek"] = df["InvoiceDate"].dt.day_name()
    df["Hour"] = df["InvoiceDate"].dt.hour
    df["Date"] = df["InvoiceDate"].dt.date
    df["WeekOfYear"] = df["InvoiceDate"].dt.isocalendar().week.astype(int)

    df["InvoiceNo"] = df["InvoiceNo"].astype(str).str.strip()
    df["StockCode"] = df["StockCode"].astype(str).str.strip()
    df["Description"] = df["Description"].astype(str).str.strip()
    df["Country"] = df["Country"].astype(str).str.strip()

    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce")

    # Drop rows where the invoice date itself is missing – they are unusable
    before = len(df)
    df = df.dropna(subset=["InvoiceDate"])
    if before - len(df):
        logger.info("Dropped %s rows with missing InvoiceDate", before - len(df))

    logger.info("Structural cleansing done. Shape: %s", df.shape)
    return df


def transactional_validation(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the data into four clean buckets:

    1. registered  – real sales with a CustomerID
    2. guests      – sales without CustomerID (still real revenue)
    3. cancellations – InvoiceNo starts with 'C'
    4. anomalies   – zero / negative price or quantity on non-cancel rows
    """
    df = df.copy()

    is_cancel = df["InvoiceNo"].str.upper().str.startswith("C")
    cancellations = df[is_cancel].copy()
    sales = df[~is_cancel].copy()

    anomaly_mask = (sales["UnitPrice"] <= 0) | (sales["Quantity"] <= 0)
    anomalies = sales[anomaly_mask].copy()
    clean_sales = sales[~anomaly_mask].copy()

    guest_mask = clean_sales["CustomerID"].isna()
    guests = clean_sales[guest_mask].copy()
    registered = clean_sales[~guest_mask].copy()

    registered["CustomerID"] = registered["CustomerID"].astype(int)

    logger.info(
        "Split → registered=%s | guests=%s | cancels=%s | anomalies=%s",
        len(registered),
        len(guests),
        len(cancellations),
        len(anomalies),
    )
    return registered, guests, cancellations, anomalies


def feature_generation(df: pd.DataFrame) -> pd.DataFrame:
    """Add LineRevenue and the UK flag used later for local vs international views."""
    df = df.copy()
    df["LineRevenue"] = df["Quantity"] * df["UnitPrice"]
    df["Is_UK"] = (df["Country"] == "United Kingdom").astype(int)
    return df


def full_pipeline(raw_path: str | Path) -> Dict[str, pd.DataFrame]:
    """
    Run the whole cleaning chain and return every intermediate frame.
    This makes every decision auditable.
    """
    raw = load_raw(raw_path)
    structured = structural_cleansing(raw)
    registered, guests, cancellations, anomalies = transactional_validation(structured)

    registered = feature_generation(registered)
    guests = feature_generation(guests)
    cancellations = feature_generation(cancellations)
    anomalies = feature_generation(anomalies)

    return {
        "raw": raw,
        "structured": structured,
        "registered": registered,
        "guests": guests,
        "cancellations": cancellations,
        "anomalies": anomalies,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = full_pipeline("data/Online_Retail.xlsx")
    for k, v in result.items():
        print(f"{k:15s}: {v.shape}")
