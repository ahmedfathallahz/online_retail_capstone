"""
Customer clustering on RFM features.

Pipeline:
1. Log1p on Monetary & Frequency (heavy right skew from wholesale accounts)
2. StandardScaler so Recency is not dominated by the monetary scale
3. K-Means (primary), Hierarchical and DBSCAN for comparison
4. Elbow + Silhouette to justify the chosen k
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def preprocess_for_clustering(
    rfm: pd.DataFrame,
    features: list[str] | None = None,
) -> Tuple[np.ndarray, StandardScaler, pd.DataFrame]:
    """
    Log-transform skewed columns then scale to zero mean / unit variance.
    Returns the scaled matrix, the fitted scaler, and the log-transformed frame.
    """
    if features is None:
        features = ["Recency", "Frequency", "Monetary"]

    X = rfm[features].copy()

    for col in ["Monetary", "Frequency"]:
        if col in X.columns:
            X[col] = np.log1p(X[col])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    logger.info("Pre-processed %s customers on features %s", X_scaled.shape[0], features)
    return X_scaled, scaler, X


def run_kmeans(
    X_scaled: np.ndarray,
    n_clusters: int = 4,
    random_state: int = 42,
) -> Tuple[np.ndarray, KMeans]:
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=20)
    labels = model.fit_predict(X_scaled)
    return labels, model


def elbow_method(
    X_scaled: np.ndarray,
    k_range: range = range(2, 11),
    random_state: int = 42,
) -> Dict[int, float]:
    """Within-cluster sum of squares for each k – used for the elbow plot."""
    wcss = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        km.fit(X_scaled)
        wcss[k] = km.inertia_
    return wcss


def silhouette_analysis(
    X_scaled: np.ndarray,
    k_range: range = range(2, 11),
    random_state: int = 42,
) -> Dict[int, float]:
    """Average silhouette score for each k."""
    scores = {}
    for k in k_range:
        labels = KMeans(n_clusters=k, random_state=random_state, n_init=10).fit_predict(
            X_scaled
        )
        scores[k] = silhouette_score(X_scaled, labels)
    return scores


def run_hierarchical(
    X_scaled: np.ndarray,
    n_clusters: int = 4,
) -> np.ndarray:
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    return model.fit_predict(X_scaled)


def run_dbscan(
    X_scaled: np.ndarray,
    eps: float = 0.8,
    min_samples: int = 10,
) -> np.ndarray:
    """Useful for isolating extreme wholesale outliers."""
    model = DBSCAN(eps=eps, min_samples=min_samples)
    return model.fit_predict(X_scaled)


def attach_labels(
    rfm: pd.DataFrame,
    labels: np.ndarray,
    col_name: str = "Cluster",
) -> pd.DataFrame:
    out = rfm.copy()
    out[col_name] = labels
    return out


def cluster_profiles(rfm: pd.DataFrame, cluster_col: str = "Cluster_KMeans") -> pd.DataFrame:
    """Quick summary table of mean RFM values per cluster."""
    return (
        rfm.groupby(cluster_col)
        .agg(
            N=("CustomerID", "count"),
            Avg_Recency=("Recency", "mean"),
            Avg_Frequency=("Frequency", "mean"),
            Avg_Monetary=("Monetary", "mean"),
            Total_Revenue=("Monetary", "sum"),
        )
        .round(1)
        .sort_values("Total_Revenue", ascending=False)
        .reset_index()
    )
