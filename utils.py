from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from gc import collect
from typing import List

import numpy as np
import torch
from matplotlib import pyplot as plt
from pandas import DataFrame, Series
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tqdm.auto import tqdm


def _collect():
    x = 0
    for _ in range(3):
        x += collect()
        torch.cuda.empty_cache()
    return x


def create_timeseries(X, cat, y, window=10, n_out=1, shift=0):
    _X, _cat, _y = [], [], []
    for i in range(len(X) - (window + n_out + shift)):
        _X.append(X[i : i + window])
        _cat.append(cat[i])
        _y.append(y[i + window + shift : i + window + shift + n_out])
    return np.asarray(_X), np.asarray(_cat), np.asarray(_y)


def remove_outliers(data: Series, n=3) -> tuple[Series, bool]:
    # Calculate Z-scores
    mean = data.mean()
    std = data.std()
    data_zscore = (data - mean) / std

    # Identify outliers (e.g., |Z| > 3)
    outliers_z = data[abs(data_zscore) > n]
    has_outliers = len(outliers_z) > 0
    clip_data = data
    if has_outliers:
        clip_data = data.clip(upper=mean + n * std, lower=mean - n * std)

    return clip_data, has_outliers


def calculate_time_features(series: DataFrame, label="present", window=1):
    if label == "present":
        prefix = "p"
        calendar = series.index
    else:
        prefix = "f"
        calendar = series.index + timedelta(days=window * 7)

    week_of_year = calendar.isocalendar().week.values
    month_of_year = calendar.month.values

    week_max = 53
    month_max = 12

    # Week features
    series[f"{prefix}_t_sin"] = np.sin(week_of_year * (2 * np.pi / week_max))
    series[f"{prefix}_t_cos"] = np.cos(week_of_year * (2 * np.pi / week_max))

    # Month features
    series[f"{prefix}_m_sin"] = np.sin(month_of_year * (2 * np.pi / month_max))
    series[f"{prefix}_m_cos"] = np.cos(month_of_year * (2 * np.pi / month_max))


# series_features = [
#     "qty_scaled",
#     "sku_code",
#     "p_t_sin",
#     "p_t_cos",
#     "p_m_sin",
#     "p_m_cos",
#     "f_t_sin",
#     "f_t_cos",
#     "f_m_sin",
#     "f_m_cos",
# ]
# series_features.extend(
#     [
#         c
#         for c in series.select_dtypes(np.number)
#         if c.startswith("qty_") and c not in set(series_features)
#     ]
# )


def plot_model(model, batch):
    from torchviz import make_dot

    qty = batch["qty"]
    past_time = batch["past_time"]
    future_time = batch["future_time"]
    sku = batch["sku"]
    cats = {key: value.to(dtype=torch.int32) for key, value in batch["cats"].items()}

    # Forward pass return both regression and classification
    with torch.no_grad():
        model.eval()
        outputs = model("0", qty, past_time, future_time, sku, cats)
    model.train()
    return make_dot(outputs, params=dict(model.named_parameters()))


def find_best_k(series, k_range=range(2, 30), random_state=42):
    inertia_values = []
    dbi_scores = []

    if series.max() > 1:
        scaler = MinMaxScaler()
        data = scaler.fit_transform(series)
    else:
        data = series

    for k in tqdm(k_range):
        kmeans = KMeans(n_clusters=k, random_state=random_state, max_iter=1000, tol=1e6)
        kmeans.fit(data)
        inertia_values.append(kmeans.inertia_)
        dbi_scores.append(davies_bouldin_score(data, kmeans.labels_))

    # Determine the "elbow point" for inertia
    deltas = np.diff(inertia_values)
    elbow_point = np.argmin(np.diff(deltas)) + k_range.start  # Adjust for indexing

    # Plot the Elbow Method and DBI
    _, ax1 = plt.subplots(figsize=(10, 6))

    # Inertia plot
    ax1.plot(k_range, inertia_values, marker="o", label="Inertia", color="blue")
    ax1.set_xlabel("Number of Clusters (K)")
    ax1.set_ylabel("Inertia", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")

    # DBI plot
    ax2 = ax1.twinx()
    ax2.plot(k_range, dbi_scores, marker="x", label="DBI", color="green")
    ax2.set_ylabel("Davies-Bouldin Index", color="green")
    ax2.tick_params(axis="y", labelcolor="green")

    # Highlight the optimal K
    ax1.axvline(
        x=elbow_point, color="red", linestyle="--", label=f"Optimal K = {elbow_point}"
    )
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")

    plt.title("Optimal K Selection: Elbow Method & DBI")
    plt.grid()
    plt.show()

    return {
        "kmeans": KMeans(
            n_clusters=elbow_point, random_state=random_state, max_iter=1000, tol=1e6
        ),
        "best_k": elbow_point,
        "inertia_values": inertia_values,
        "dbi_scores": dbi_scores,
    }
