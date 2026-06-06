"""
K-EDA — Dimensionality Reduction Engine
=============================================
PCA, clustering, Hopkins statistic, and density analysis.
"""

import numpy as np
import pandas as pd


def pca_analysis(df, numeric_cols, n_components=None):
    """
    Perform PCA and return variance explained information.

    Returns
    -------
    dict : {explained_variance_ratio, cumulative_variance, components_data, n_components}
    """
    try:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return {"error": "scikit-learn not installed"}

    if len(numeric_cols) < 2:
        return {"error": "Need at least 2 numeric columns for PCA"}

    data = df[numeric_cols].copy()
    data = data.fillna(data.median())

    # Sample for performance if large
    if len(data) > 20000:
        data = data.sample(20000, random_state=42)

    # Standardize
    scaler = StandardScaler()
    scaled = scaler.fit_transform(data)

    # Determine components
    max_comp = min(len(numeric_cols), len(data), 20)
    if n_components is None:
        n_components = max_comp

    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(scaled)

    cum_var = np.cumsum(pca.explained_variance_ratio_)
    # Components to reach 95% variance
    n_95 = int(np.argmax(cum_var >= 0.95)) + 1 if any(cum_var >= 0.95) else n_components

    return {
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "cumulative_variance": cum_var.tolist(),
        "n_components_95": n_95,
        "n_features": len(numeric_cols),
        "components_2d": [],  # Removed list-of-lists conversion since it is unused
    }


def kmeans_preview(df, numeric_cols, n_clusters=3):
    """
    Perform K-Means clustering on PCA-reduced data for preview.

    Returns
    -------
    dict : {labels, centers, inertia, pca_data}
    """
    try:
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return {"error": "scikit-learn not installed"}

    if len(numeric_cols) < 2:
        return {"error": "Need at least 2 numeric columns"}

    data = df[numeric_cols].copy()
    data = data.fillna(data.median())

    # Sample for performance
    if len(data) > 5000:
        data = data.sample(5000, random_state=42)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(data)

    # PCA to 2D
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(scaled)

    # K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(pca_data)

    return {
        "labels": labels.tolist(),
        "pca_data": pca_data.tolist(),
        "inertia": float(kmeans.inertia_),
        "n_clusters": n_clusters,
    }


def hopkins_statistic(df, numeric_cols, sample_size=None):
    """
    Compute Hopkins statistic to assess cluster tendency.
    Values close to 0.5 indicate random data, close to 1 indicate clustered data.

    Returns
    -------
    float : Hopkins statistic (0 to 1)
    """
    if len(numeric_cols) < 2:
        return 0.5

    data = df[numeric_cols].copy()
    data = data.fillna(data.median()).values

    n = len(data)
    if sample_size is None:
        sample_size = min(n, 1000)

    if n < 10:
        return 0.5

    try:
        from sklearn.neighbors import NearestNeighbors

        # Random sample from data
        rng = np.random.RandomState(42)
        sample_indices = rng.choice(n, sample_size, replace=False)
        sample = data[sample_indices]

        # Random uniform points in the data space
        mins = data.min(axis=0)
        maxs = data.max(axis=0)
        random_points = rng.uniform(mins, maxs, (sample_size, data.shape[1]))

        # Nearest neighbor distances
        nn = NearestNeighbors(n_neighbors=2)
        nn.fit(data)

        # Distance from sample points to nearest neighbor in data
        u_dist, _ = nn.kneighbors(random_points, n_neighbors=1)
        w_dist, _ = nn.kneighbors(sample, n_neighbors=2)

        u_sum = np.sum(u_dist)
        w_sum = np.sum(w_dist[:, 1])  # Second neighbor (first is itself)

        if u_sum + w_sum == 0:
            return 0.5

        return float(u_sum / (u_sum + w_sum))

    except Exception:
        return 0.5
