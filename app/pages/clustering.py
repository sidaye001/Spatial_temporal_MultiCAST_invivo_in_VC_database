import os
from functools import lru_cache

import numpy as np
import pandas as pd
import dash
from dash import html, dcc, dash_table, Input, Output, State, no_update
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster
import plotly.express as px

try:
    from statsmodels.nonparametric.smoothers_lowess import lowess
    HAS_STATSMODELS = True
except Exception:
    HAS_STATSMODELS = False


dash.register_page(__name__, path="/clustering", name="Clustering")


# ============================================================
# 0. Paths and constants
# ============================================================

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(APP_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

RAW_FILE = os.path.join(
    DATA_DIR,
    "raw",
    "Spatial_temporal_MultiSCAST_FC_final_capping.csv",
)

ANNOTATION_FILE = os.path.join(
    DATA_DIR,
    "annotation",
    "new_annotations_with_uniprot_names.csv",
)

PRECOMPUTED_CLUSTER_FILE = os.path.join(
    DATA_DIR,
    "clustering",
    "invivo_DTW_cosine_clustering_3Dlevel.csv",
)

SPACE_LEVELS = [
    "st", "SI1", "SI2", "SI3", "SI4", "SI5",
    "SI6", "SI7", "SI8", "SI9", "ce", "co",
]

TIME_LEVELS = ["1h", "3h", "6h", "12h", "24h"]

TIME_NUM_MAP = {"1h": 1, "3h": 3, "6h": 6, "12h": 12, "24h": 24}
SPACE_NUM_MAP = {s: i + 1 for i, s in enumerate(SPACE_LEVELS)}

DEFAULT_K = 20
DEFAULT_CLUSTERING_MODE = "global"
DEFAULT_CLUSTERING_METHOD = "dtw"
DEFAULT_VISUALIZATION_MODE = "spatial_profile"

COLOR_POOL = (
    px.colors.qualitative.Plotly
    + px.colors.qualitative.Dark24
    + px.colors.qualitative.Light24
    + px.colors.qualitative.Set3
    + px.colors.qualitative.Alphabet
)


# ============================================================
# 1. Data loading
# ============================================================

@lru_cache(maxsize=1)
def load_raw_data():
    if not os.path.exists(RAW_FILE):
        raise FileNotFoundError(f"Raw fitness file not found: {RAW_FILE}")

    df = pd.read_csv(RAW_FILE)

    required_cols = {"Gene", "Time", "Space"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in raw data: {missing}")

    if "logFC" not in df.columns:
        if "Beta" in df.columns:
            df = df.rename(columns={"Beta": "logFC"})
        else:
            raise ValueError("Raw data must contain either 'logFC' or 'Beta'.")

    df = df[["Gene", "Time", "Space", "logFC"]].copy()
    df["Gene"] = df["Gene"].astype(str).str.strip()
    df["Time"] = df["Time"].astype(str).str.strip()
    df["Space"] = df["Space"].astype(str).str.strip()
    df["logFC"] = pd.to_numeric(df["logFC"], errors="coerce")

    df = df[df["Time"].isin(TIME_LEVELS)].copy()
    df = df[df["Space"].isin(SPACE_LEVELS)].copy()
    df = df.dropna(subset=["Gene", "Time", "Space", "logFC"]).copy()

    df["Time"] = pd.Categorical(df["Time"], categories=TIME_LEVELS, ordered=True)
    df["Space"] = pd.Categorical(df["Space"], categories=SPACE_LEVELS, ordered=True)
    df["TimeNum"] = df["Time"].astype(str).map(TIME_NUM_MAP)
    df["SpaceNum"] = df["Space"].astype(str).map(SPACE_NUM_MAP)

    return df


@lru_cache(maxsize=1)
def load_precomputed_cluster_data():
    """Load the precomputed 3D global clustering table."""
    if not os.path.exists(PRECOMPUTED_CLUSTER_FILE):
        return None

    df = pd.read_csv(PRECOMPUTED_CLUSTER_FILE)

    required_cols = {"Gene", "Time", "Space"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Precomputed clustering file is missing required columns: {missing}. "
            f"File path: {PRECOMPUTED_CLUSTER_FILE}"
        )

    if "logFC" not in df.columns:
        if "Beta" in df.columns:
            df = df.rename(columns={"Beta": "logFC"})
        else:
            raise ValueError("Precomputed clustering file must contain either 'logFC' or 'Beta'.")

    df = df.copy()
    df["Gene"] = df["Gene"].astype(str).str.strip()
    df["Time"] = df["Time"].astype(str).str.strip()
    df["Space"] = df["Space"].astype(str).str.strip()
    df["logFC"] = pd.to_numeric(df["logFC"], errors="coerce")

    df = df[df["Time"].isin(TIME_LEVELS)].copy()
    df = df[df["Space"].isin(SPACE_LEVELS)].copy()
    df = df.dropna(subset=["Gene", "Time", "Space", "logFC"]).copy()

    df["Time"] = pd.Categorical(df["Time"], categories=TIME_LEVELS, ordered=True)
    df["Space"] = pd.Categorical(df["Space"], categories=SPACE_LEVELS, ordered=True)
    df["TimeNum"] = df["Time"].astype(str).map(TIME_NUM_MAP)
    df["SpaceNum"] = df["Space"].astype(str).map(SPACE_NUM_MAP)

    return df


@lru_cache(maxsize=1)
def load_annotation():
    raw = load_raw_data()
    genes = sorted(raw["Gene"].unique())

    if not os.path.exists(ANNOTATION_FILE):
        return pd.DataFrame({"Gene": genes, "GeneName": genes})

    ann = pd.read_csv(ANNOTATION_FILE)
    if "locus_ID" not in ann.columns:
        return pd.DataFrame({"Gene": genes, "GeneName": genes})

    if "gene_name" not in ann.columns:
        ann["gene_name"] = ""

    ann = ann[["locus_ID", "gene_name"]].copy()
    ann["locus_ID"] = ann["locus_ID"].astype(str).str.strip()
    ann["gene_name"] = ann["gene_name"].fillna("").astype(str).str.strip()
    ann["GeneName"] = np.where(ann["gene_name"] != "", ann["gene_name"], ann["locus_ID"])

    ann = (
        ann.rename(columns={"locus_ID": "Gene"})
        [["Gene", "GeneName"]]
        .drop_duplicates()
    )
    return ann


def add_annotation(df):
    ann = load_annotation().copy()
    out = df.copy()

    drop_cols = [
        c for c in ["GeneName", "gene_name", "GeneName_x", "GeneName_y", "gene_name_x", "gene_name_y"]
        if c in out.columns
    ]
    if drop_cols:
        out = out.drop(columns=drop_cols)

    out["Gene"] = out["Gene"].astype(str)
    ann["Gene"] = ann["Gene"].astype(str)
    out = out.merge(ann[["Gene", "GeneName"]], on="Gene", how="left")
    out["GeneName"] = out["GeneName"].fillna(out["Gene"])
    return out


# ============================================================
# 2. Utility helpers
# ============================================================

def safe_int(value, default=20, minimum=2, maximum=100):
    try:
        if value is None or value == "":
            out = int(default)
        else:
            out = int(value)
    except Exception:
        out = int(default)

    out = max(int(minimum), out)
    out = min(int(maximum), out)
    return out


def safe_float(value, default=0.8, minimum=0.1, maximum=1.0):
    try:
        if value is None or value == "":
            out = float(default)
        else:
            out = float(value)
    except Exception:
        out = float(default)

    out = max(float(minimum), out)
    out = min(float(maximum), out)
    return out


def ordered_range(levels, start_value, end_value):
    if start_value not in levels:
        start_value = levels[0]
    if end_value not in levels:
        end_value = levels[-1]

    i = levels.index(start_value)
    j = levels.index(end_value)
    if i <= j:
        return levels[i:j + 1]
    return levels[j:i + 1]


def find_first_existing_column(df, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    return None


def get_precomputed_dtw_cluster_column():
    pre = load_precomputed_cluster_data()
    if pre is None:
        return None
    return find_first_existing_column(pre, ["Cluster_DTW", "cluster_DTW", "Cluster_dtw", "cluster_dtw"])


def get_precomputed_cosine_cluster_column():
    pre = load_precomputed_cluster_data()
    if pre is None:
        return None
    return find_first_existing_column(pre, ["Cluster_cosine", "cluster_cosine", "Cluster_Cosine", "cluster_Cosine"])


def get_precomputed_dtw_silhouette_column():
    pre = load_precomputed_cluster_data()
    if pre is None:
        return None
    return find_first_existing_column(
        pre,
        ["Silhouette_DTW", "silhouette_DTW", "Silhouette_dtw", "silhouette_dtw", "sil_DTW", "sil_dtw"],
    )


def get_precomputed_cosine_silhouette_column():
    pre = load_precomputed_cluster_data()
    if pre is None:
        return None
    return find_first_existing_column(
        pre,
        ["Silhouette_cosine", "silhouette_cosine", "Silhouette_Cosine", "silhouette_Cosine", "sil_cosine", "sil_Cosine"],
    )


def zscore_rows(X):
    X = X.astype(float)
    mean = np.nanmean(X, axis=1, keepdims=True)
    sd = np.nanstd(X, axis=1, keepdims=True)
    sd[~np.isfinite(sd)] = 1.0
    sd[sd == 0] = 1.0
    return (X - mean) / sd


# ============================================================
# 3. Precomputed Module 1 result
# ============================================================

def get_precomputed_result_for_module1():
    pre = load_precomputed_cluster_data()
    if pre is None:
        raise FileNotFoundError(
            f"Precomputed clustering file not found: {PRECOMPUTED_CLUSTER_FILE}"
        )

    dtw_col = get_precomputed_dtw_cluster_column()
    cosine_col = get_precomputed_cosine_cluster_column()
    if dtw_col is None:
        raise ValueError("Cannot find DTW cluster column in precomputed table. Expected Cluster_DTW.")

    dtw_sil_col = get_precomputed_dtw_silhouette_column()
    cosine_sil_col = get_precomputed_cosine_silhouette_column()

    all_tb = pre.copy()
    all_tb["Cluster_DTW"] = pd.to_numeric(all_tb[dtw_col], errors="coerce")

    if cosine_col is not None:
        all_tb["Cluster_cosine"] = pd.to_numeric(all_tb[cosine_col], errors="coerce")
    else:
        all_tb["Cluster_cosine"] = np.nan

    if dtw_sil_col is not None:
        all_tb["Silhouette_DTW"] = pd.to_numeric(all_tb[dtw_sil_col], errors="coerce")
    else:
        all_tb["Silhouette_DTW"] = np.nan

    if cosine_sil_col is not None:
        all_tb["Silhouette_cosine"] = pd.to_numeric(all_tb[cosine_sil_col], errors="coerce")
    else:
        all_tb["Silhouette_cosine"] = np.nan

    all_tb = all_tb.dropna(subset=["Cluster_DTW"]).copy()
    all_tb["Cluster_DTW"] = all_tb["Cluster_DTW"].astype(int)

    if "Cluster_cosine" in all_tb.columns and all_tb["Cluster_cosine"].notna().any():
        all_tb["Cluster_cosine"] = all_tb["Cluster_cosine"].astype("Int64")

    all_tb = add_annotation(all_tb)

    gene_clusters = (
        all_tb[["Gene", "GeneName", "Cluster_DTW", "Silhouette_DTW", "Cluster_cosine", "Silhouette_cosine"]]
        .drop_duplicates()
        .sort_values(["Cluster_DTW", "GeneName", "Gene"])
        .reset_index(drop=True)
    )

    gene_clusters["Cluster"] = gene_clusters["Cluster_DTW"]
    gene_clusters["Silhouette"] = gene_clusters["Silhouette_DTW"]
    gene_clusters["Active_Clustering_Method"] = "DTW distance"

    all_tb["Cluster"] = all_tb["Cluster_DTW"]
    all_tb["Silhouette"] = all_tb["Silhouette_DTW"]
    all_tb["Active_Clustering_Method"] = "DTW distance"

    return gene_clusters, all_tb


# ============================================================
# 4. Custom clustering logic for Module 2
# ============================================================

def build_custom_feature_matrix(mode, start_time, end_time, start_space, end_space, scale_per_gene):
    """
    Build the gene-feature matrix for custom clustering.

    mode='global': each gene is represented by all selected time x space values.
    mode='spatial': each gene is represented by selected spatial values averaged over the selected time window.
    mode='temporal': each gene is represented by selected time values averaged over the selected space window.
    """
    selected_times = ordered_range(TIME_LEVELS, start_time, end_time)
    selected_spaces = ordered_range(SPACE_LEVELS, start_space, end_space)

    df = load_raw_data().copy()
    genes = sorted(df["Gene"].unique())
    df = df[df["Time"].astype(str).isin(selected_times)].copy()
    df = df[df["Space"].astype(str).isin(selected_spaces)].copy()

    if df.empty:
        raise ValueError("No data found in the selected time/space window.")

    if mode == "spatial":
        complete_index = pd.MultiIndex.from_product([genes, selected_spaces], names=["Gene", "Space"])
        feature_df = (
            df.groupby(["Gene", "Space"], observed=False)["logFC"]
            .mean()
            .reindex(complete_index)
            .reset_index()
        )
        feature_df["logFC"] = feature_df["logFC"].fillna(0)
        Xdf = (
            feature_df.pivot(index="Gene", columns="Space", values="logFC")
            .reindex(index=genes, columns=selected_spaces)
            .fillna(0)
        )
        feature_labels = selected_spaces

    elif mode == "temporal":
        complete_index = pd.MultiIndex.from_product([genes, selected_times], names=["Gene", "Time"])
        feature_df = (
            df.groupby(["Gene", "Time"], observed=False)["logFC"]
            .mean()
            .reindex(complete_index)
            .reset_index()
        )
        feature_df["logFC"] = feature_df["logFC"].fillna(0)
        Xdf = (
            feature_df.pivot(index="Gene", columns="Time", values="logFC")
            .reindex(index=genes, columns=selected_times)
            .fillna(0)
        )
        feature_labels = selected_times

    else:
        complete_index = pd.MultiIndex.from_product(
            [genes, selected_times, selected_spaces], names=["Gene", "Time", "Space"]
        )
        feature_df = (
            df[["Gene", "Time", "Space", "logFC"]]
            .set_index(["Gene", "Time", "Space"])
            .reindex(complete_index)
            .reset_index()
        )
        feature_df["logFC"] = feature_df["logFC"].fillna(0)
        feature_df["Feature"] = feature_df["Time"].astype(str) + "_" + feature_df["Space"].astype(str)
        feature_order = [f"{t}_{s}" for t in selected_times for s in selected_spaces]
        Xdf = (
            feature_df.pivot(index="Gene", columns="Feature", values="logFC")
            .reindex(index=genes, columns=feature_order)
            .fillna(0)
        )
        feature_labels = feature_order

    X = Xdf.values.astype(float)
    if scale_per_gene:
        X = zscore_rows(X)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    return Xdf.index.tolist(), X, feature_labels, selected_times, selected_spaces


def cosine_distance_vectorized(X):
    Dvec = pdist(X, metric="cosine")
    return np.nan_to_num(Dvec, nan=0.0, posinf=0.0, neginf=0.0)


def dtw_distance_1d(x, y, window=None):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    m = len(y)

    if window is None:
        window = max(n, m)
    else:
        window = max(int(window), abs(n - m))

    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0.0

    for i in range(1, n + 1):
        j_start = max(1, i - window)
        j_end = min(m, i + window)
        for j in range(j_start, j_end + 1):
            cost = (x[i - 1] - y[j - 1]) ** 2
            dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])

    return float(np.sqrt(dtw[n, m]))


def dtw_distance_vector(X, window_fraction=0.05):
    n = X.shape[0]
    length = X.shape[1]
    window = max(1, int(round(window_fraction * length)))

    out = []
    for i in range(n - 1):
        xi = X[i, :]
        for j in range(i + 1, n):
            out.append(dtw_distance_1d(xi, X[j, :], window=window))

    Dvec = np.array(out, dtype=float)
    return np.nan_to_num(Dvec, nan=0.0, posinf=0.0, neginf=0.0)


def compute_silhouette_from_distance(D, labels):
    labels = np.asarray(labels)
    n = len(labels)
    if n <= 2:
        return np.zeros(n)

    unique_clusters = np.unique(labels)
    if len(unique_clusters) <= 1:
        return np.zeros(n)

    sil = np.zeros(n)
    for i in range(n):
        same = labels == labels[i]
        same[i] = False
        a = np.mean(D[i, same]) if np.any(same) else 0.0

        b_vals = []
        for c in unique_clusters:
            if c == labels[i]:
                continue
            other = labels == c
            if np.any(other):
                b_vals.append(np.mean(D[i, other]))

        b = min(b_vals) if b_vals else 0.0
        denom = max(a, b)
        sil[i] = 0.0 if denom == 0 else (b - a) / denom

    return sil


def cluster_from_distance_vector(Dvec, k):
    n = int((1 + np.sqrt(1 + 8 * len(Dvec))) / 2)
    if n <= 1:
        return np.ones(n, dtype=int), np.zeros(n)

    if np.all(Dvec == 0):
        return np.ones(n, dtype=int), np.zeros(n)

    Z = linkage(Dvec, method="average")
    labels = fcluster(Z, t=int(k), criterion="maxclust").astype(int)
    D = squareform(Dvec)
    sil = compute_silhouette_from_distance(D, labels)
    return labels, sil


@lru_cache(maxsize=32)
def compute_custom_clustering_cached(
    mode,
    k,
    method,
    scale_per_gene,
    start_time,
    end_time,
    start_space,
    end_space,
):
    genes, X, feature_labels, selected_times, selected_spaces = build_custom_feature_matrix(
        mode=mode,
        start_time=start_time,
        end_time=end_time,
        start_space=start_space,
        end_space=end_space,
        scale_per_gene=scale_per_gene,
    )

    n = len(genes)
    k = safe_int(k, default=20, minimum=2, maximum=max(2, n))

    if method == "dtw":
        Dvec = dtw_distance_vector(X, window_fraction=0.05)
        cluster_col = "Cluster_DTW"
        sil_col = "Silhouette_DTW"
        active_method = "DTW distance"
    else:
        Dvec = cosine_distance_vectorized(X)
        cluster_col = "Cluster_cosine"
        sil_col = "Silhouette_cosine"
        active_method = "Cosine distance"

    labels, sil = cluster_from_distance_vector(Dvec, k)

    gene_clusters = pd.DataFrame({"Gene": genes, cluster_col: labels, sil_col: sil})
    if method == "dtw":
        gene_clusters["Cluster_cosine"] = np.nan
        gene_clusters["Silhouette_cosine"] = np.nan
    else:
        gene_clusters["Cluster_DTW"] = np.nan
        gene_clusters["Silhouette_DTW"] = np.nan

    gene_clusters = add_annotation(gene_clusters)
    gene_clusters["Cluster"] = gene_clusters[cluster_col]
    gene_clusters["Silhouette"] = gene_clusters[sil_col]
    gene_clusters["Active_Clustering_Method"] = active_method
    gene_clusters["Custom_Clustering_Mode"] = mode
    gene_clusters["Feature_Count"] = len(feature_labels)
    gene_clusters["Selected_Times"] = ", ".join(selected_times)
    gene_clusters["Selected_Spaces"] = ", ".join(selected_spaces)

    raw = load_raw_data().copy()
    all_tb = raw.merge(
        gene_clusters[["Gene", "GeneName", "Cluster", "Silhouette", "Active_Clustering_Method"]],
        on="Gene",
        how="left",
    )
    all_tb["Custom_Clustering_Mode"] = mode
    all_tb["Selected_Times"] = ", ".join(selected_times)
    all_tb["Selected_Spaces"] = ", ".join(selected_spaces)

    return gene_clusters, all_tb, feature_labels, selected_times, selected_spaces


# ============================================================
# 5. Plot helpers
# ============================================================

def filter_to_selected_clusters(gene_clusters, all_tb, selected_clusters):
    if not selected_clusters:
        return gene_clusters.copy(), all_tb.copy()

    selected_clusters = [int(x) for x in selected_clusters]
    gc = gene_clusters[gene_clusters["Cluster"].isin(selected_clusters)].copy()
    tb = all_tb[all_tb["Cluster"].isin(selected_clusters)].copy()
    return gc, tb


def make_cluster_options(gene_clusters):
    clusters = sorted(
        pd.to_numeric(gene_clusters["Cluster"], errors="coerce")
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )
    return [{"label": f"Cluster {c}", "value": c} for c in clusters]


def fit_curve(x_num, y, fit_method="loess", span=0.8):
    x_num = np.asarray(x_num, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x_num) & np.isfinite(y)
    if ok.sum() < 2:
        return x_num[ok], y[ok]

    x_ok = x_num[ok]
    y_ok = y[ok]
    order = np.argsort(x_ok)
    x_ok = x_ok[order]
    y_ok = y_ok[order]

    if fit_method == "none":
        return x_ok, y_ok

    x_grid = np.linspace(np.min(x_ok), np.max(x_ok), 200)

    if fit_method == "loess":
        if HAS_STATSMODELS and len(np.unique(x_ok)) >= 3:
            sm = lowess(y_ok, x_ok, frac=float(span), return_sorted=True)
            tmp = pd.DataFrame({"x": sm[:, 0], "y": sm[:, 1]})
            tmp = tmp.groupby("x", as_index=False)["y"].mean().sort_values("x")
            if tmp.shape[0] >= 2:
                return x_grid, np.interp(x_grid, tmp["x"], tmp["y"])
        return x_ok, y_ok

    if fit_method in ["poly2", "poly3"]:
        degree = 2 if fit_method == "poly2" else 3
        if len(np.unique(x_ok)) < degree + 1:
            return x_ok, y_ok
        try:
            coef = np.polyfit(x_ok, y_ok, deg=degree)
            return x_grid, np.polyval(coef, x_grid)
        except Exception:
            return x_ok, y_ok

    return x_ok, y_ok


def gene_color_map(genes):
    genes = sorted(list(set(genes)))
    return {gene: COLOR_POOL[i % len(COLOR_POOL)] for i, gene in enumerate(genes)}


def make_faceted_profile_figure(
    all_tb,
    visualization_mode,
    visualization_time,
    visualization_space,
    max_genes_per_cluster,
    show_median=False,
    curve_fit_method="none",
    loess_span=0.8,
    fixed_y_axis=False,
    individual_curve_style="grey",
    display_times=None,
    display_spaces=None,
):
    df = all_tb.copy()
    if "GeneName" not in df.columns:
        df = add_annotation(df)

    df["Time"] = df["Time"].astype(str).str.strip()
    df["Space"] = df["Space"].astype(str).str.strip()
    display_times = [t for t in (display_times or TIME_LEVELS) if t in TIME_LEVELS]
    display_spaces = [s for s in (display_spaces or SPACE_LEVELS) if s in SPACE_LEVELS]
    if not display_times:
        display_times = TIME_LEVELS.copy()
    if not display_spaces:
        display_spaces = SPACE_LEVELS.copy()

    df = df[df["Time"].isin(display_times)].copy()
    df = df[df["Space"].isin(display_spaces)].copy()

    df["Time"] = pd.Categorical(df["Time"], categories=display_times, ordered=True)
    df["Space"] = pd.Categorical(df["Space"], categories=display_spaces, ordered=True)
    df["TimeNum"] = df["Time"].astype(str).map(TIME_NUM_MAP)
    df["SpaceNum"] = df["Space"].astype(str).map(SPACE_NUM_MAP)

    if visualization_mode == "spatial_profile":
        if visualization_time not in display_times:
            visualization_time = display_times[0]
        df = df[df["Time"].astype(str) == visualization_time].copy()
        x_cat = "Space"
        x_num = "SpaceNum"
        x_order = display_spaces
        x_tickvals = [SPACE_NUM_MAP[s] for s in display_spaces]
        x_ticktext = display_spaces
        x_title = "GI tract location"
        title = f"Clustered spatial fitness profiles at {visualization_time}"
    else:
        if visualization_space not in display_spaces:
            visualization_space = display_spaces[0]
        df = df[df["Space"].astype(str) == visualization_space].copy()
        x_cat = "Time"
        x_num = "TimeNum"
        x_order = display_times
        x_tickvals = [TIME_NUM_MAP[t] for t in display_times]
        x_ticktext = display_times
        x_title = "Time"
        title = f"Clustered temporal fitness profiles at {visualization_space}"

    df = df.dropna(subset=["Cluster", "Gene", "logFC"]).copy()
    df["Cluster"] = pd.to_numeric(df["Cluster"], errors="coerce")
    df = df.dropna(subset=["Cluster"]).copy()
    df["Cluster"] = df["Cluster"].astype(int)

    clusters = sorted(df["Cluster"].dropna().unique().astype(int))
    n_clusters = len(clusters)
    if n_clusters == 0:
        fig = go.Figure()
        fig.update_layout(template="plotly_white", title="No clusters available")
        return fig

    ncols = 5 if n_clusters >= 10 else min(4, n_clusters)
    nrows = int(np.ceil(n_clusters / ncols))

    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        subplot_titles=[f"Cluster {c}" for c in clusters],
        horizontal_spacing=0.035,
        vertical_spacing=0.085,
    )

    y_min = float(df["logFC"].min())
    y_max = float(df["logFC"].max())
    if np.isfinite(y_min) and np.isfinite(y_max) and y_min != y_max:
        pad = 0.05 * (y_max - y_min)
        y_range = [y_min - pad, y_max + pad]
    else:
        y_range = None

    rng = np.random.default_rng(1)
    cmap = gene_color_map(df["Gene"].dropna().unique())
    max_genes_per_cluster = safe_int(max_genes_per_cluster, default=250, minimum=0, maximum=5000)

    for idx, c in enumerate(clusters):
        row = idx // ncols + 1
        col = idx % ncols + 1
        sub = df[df["Cluster"] == c].copy()
        genes = sorted(sub["Gene"].dropna().unique())

        if max_genes_per_cluster <= 0:
            genes_to_plot = []
        elif len(genes) > max_genes_per_cluster:
            genes_to_plot = rng.choice(genes, size=max_genes_per_cluster, replace=False)
        else:
            genes_to_plot = genes

        for gene in genes_to_plot:
            gdf = sub[sub["Gene"] == gene].sort_values(x_num)
            gname = gdf["GeneName"].iloc[0] if "GeneName" in gdf.columns and len(gdf) > 0 else gene
            sx, sy = fit_curve(gdf[x_num].values, gdf["logFC"].values, fit_method=curve_fit_method, span=loess_span)

            if individual_curve_style == "colored":
                line_color = cmap.get(gene, "rgba(0,150,200,0.9)")
                line_width = 1.8
                opacity = 0.90
            else:
                line_color = "rgba(95,95,95,0.28)"
                line_width = 1.05
                opacity = 1.0

            fig.add_trace(
                go.Scatter(
                    x=sx,
                    y=sy,
                    mode="lines",
                    line=dict(color=line_color, width=line_width, shape="spline" if curve_fit_method != "none" else "linear"),
                    opacity=opacity,
                    hovertemplate=(
                        f"GeneName: {gname}<br>"
                        f"Gene ID: {gene}<br>"
                        f"Cluster: {c}<br>"
                        f"{x_title}: %{{x}}<br>"
                        "logFC: %{y:.3f}<extra></extra>"
                    ),
                    showlegend=False,
                ),
                row=row,
                col=col,
            )

        if show_median:
            med = sub.groupby(x_cat, observed=False)["logFC"].median().reindex(x_order).reset_index()
            med[x_num] = med[x_cat].astype(str).map(SPACE_NUM_MAP if visualization_mode == "spatial_profile" else TIME_NUM_MAP)
            med = med.dropna(subset=[x_num, "logFC"]).sort_values(x_num)
            mx, my = fit_curve(med[x_num].values, med["logFC"].values, fit_method=curve_fit_method, span=loess_span)
            fig.add_trace(
                go.Scatter(
                    x=mx,
                    y=my,
                    mode="lines",
                    line=dict(color="black", width=3.3, shape="spline" if curve_fit_method != "none" else "linear"),
                    hovertemplate=(
                        f"Cluster {c}<br>"
                        f"{x_title}: %{{x}}<br>"
                        "Median logFC: %{y:.3f}<extra></extra>"
                    ),
                    name="Cluster median",
                    showlegend=(idx == 0),
                ),
                row=row,
                col=col,
            )

        fig.update_xaxes(
            tickmode="array",
            tickvals=x_tickvals,
            ticktext=x_ticktext,
            title_text=x_title if row == nrows else "",
            row=row,
            col=col,
            showgrid=True,
            gridcolor="rgba(220,220,220,0.8)",
            linecolor="rgba(80,80,80,0.75)",
            mirror=True,
        )
        fig.update_yaxes(
            title_text="logFC" if col == 1 else "",
            range=y_range if fixed_y_axis else None,
            row=row,
            col=col,
            showgrid=True,
            gridcolor="rgba(220,220,220,0.8)",
            linecolor="rgba(80,80,80,0.75)",
            mirror=True,
        )

    for ann in fig.layout.annotations:
        ann.font = dict(size=12, color="rgba(30,30,30,1)")

    fig.update_layout(
        title=title,
        template="plotly_white",
        height=max(440, nrows * 295),
        margin=dict(l=60, r=35, t=95, b=70),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(size=13),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    if visualization_mode == "spatial_profile":
        fig.update_xaxes(tickangle=90)

    return fig


def make_global_3d_cluster_figure(all_tb, display_times=None, display_spaces=None):
    df = all_tb.copy()
    if "GeneName" not in df.columns:
        df = add_annotation(df)

    df["Time"] = df["Time"].astype(str)
    df["Space"] = df["Space"].astype(str)

    display_times = [t for t in (display_times or TIME_LEVELS) if t in TIME_LEVELS]
    display_spaces = [s for s in (display_spaces or SPACE_LEVELS) if s in SPACE_LEVELS]
    if not display_times:
        display_times = TIME_LEVELS.copy()
    if not display_spaces:
        display_spaces = SPACE_LEVELS.copy()

    df = df[df["Time"].isin(display_times)].copy()
    df = df[df["Space"].isin(display_spaces)].copy()

    med = df.groupby(["Cluster", "Time", "Space"], observed=False)["logFC"].median().reset_index()
    med["TimeNum"] = med["Time"].map(TIME_NUM_MAP)
    med["SpaceNum"] = med["Space"].map(SPACE_NUM_MAP)
    med["Time"] = pd.Categorical(med["Time"], categories=TIME_LEVELS, ordered=True)
    med["Space"] = pd.Categorical(med["Space"], categories=SPACE_LEVELS, ordered=True)
    med = med.sort_values(["Cluster", "Time", "Space"])

    fig = go.Figure()
    clusters = sorted(med["Cluster"].dropna().unique().astype(int))

    for c in clusters:
        sub = med[med["Cluster"] == c].copy()
        fig.add_trace(
            go.Scatter3d(
                x=sub["TimeNum"],
                y=sub["SpaceNum"],
                z=sub["logFC"],
                mode="lines+markers",
                marker=dict(size=4),
                line=dict(width=4),
                name=f"Cluster {c}",
                text=[
                    f"Cluster {c}<br>Time: {t}<br>Space: {s}<br>Median logFC: {z:.3f}"
                    for t, s, z in zip(sub["Time"], sub["Space"], sub["logFC"])
                ],
                hoverinfo="text",
            )
        )

    fig.update_layout(
        title="3D cluster median spatial-temporal fitness profiles",
        template="plotly_white",
        height=760,
        scene=dict(
            xaxis=dict(title="Time", tickmode="array", tickvals=[TIME_NUM_MAP[t] for t in display_times], ticktext=display_times),
            yaxis=dict(title="GI tract location", tickmode="array", tickvals=[SPACE_NUM_MAP[s] for s in display_spaces], ticktext=display_spaces),
            zaxis=dict(title="Median logFC"),
            camera=dict(eye=dict(x=1.8, y=1.8, z=1.2)),
        ),
        margin=dict(l=0, r=0, t=90, b=0),
        legend=dict(itemsizing="constant"),
    )
    return fig


def summarize_result(
    gene_clusters,
    mode,
    k,
    method_label,
    source_text,
    selected_times,
    selected_spaces,
    selected_clusters,
):
    n_genes = gene_clusters["Gene"].nunique()
    n_clusters = gene_clusters["Cluster"].nunique()
    mean_sil = pd.to_numeric(gene_clusters.get("Silhouette", pd.Series(dtype=float)), errors="coerce").mean(skipna=True)

    if mode == "spatial":
        mode_text = "2D spatial clustering: gene vectors are selected GI-tract locations averaged over the selected time window"
    elif mode == "temporal":
        mode_text = "2D temporal clustering: gene vectors are selected time points averaged over the selected space window"
    else:
        mode_text = "3D global clustering: gene vectors are selected time × space values"

    cluster_text = ", ".join([str(x) for x in selected_clusters]) if selected_clusters else "All clusters"

    children = [
        html.Strong("Clustering summary"),
        html.Br(),
        f"Data source: {source_text}",
        html.Br(),
        f"Active clustering method: {method_label}",
        html.Br(),
        f"Clustering mode: {mode_text}",
        html.Br(),
        f"Selected time window: {', '.join(selected_times)}",
        html.Br(),
        f"Selected space window: {', '.join(selected_spaces)}",
        html.Br(),
        f"Displayed clusters: {cluster_text}",
        html.Br(),
        f"Number of displayed genes: {n_genes}",
        html.Br(),
        f"Number of clusters requested: {k}",
        html.Br(),
        f"Number of displayed clusters: {n_clusters}",
    ]

    if np.isfinite(mean_sil):
        children.extend([html.Br(), f"Mean silhouette score: {mean_sil:.3f}"])
    else:
        children.extend([html.Br(), "Mean silhouette score: not available"])

    return dbc.Alert(children, color="info", className="mb-3")


def gene_cluster_table_explanation():
    return dbc.Alert(
        [
            html.H5("How to interpret the gene cluster table", className="mb-2"),
            html.P(
                "The table reports gene-level cluster assignments and silhouette scores. The active Cluster "
                "and Silhouette columns correspond to the clustering method used for the current plot.",
                className="mb-2",
            ),
            html.Ul(
                [
                    html.Li([html.Strong("Cluster: "), "The active cluster assignment used for plotting and filtering."]),
                    html.Li([html.Strong("Silhouette: "), "How well a gene fits its assigned cluster. Values closer to 1 indicate better separation; values near 0 indicate boundary cases; negative values suggest poor assignment."]),
                    html.Li([html.Strong("Cluster_DTW / Cluster_cosine: "), "Method-specific cluster labels when available."]),
                ],
                className="mb-0",
            ),
        ],
        color="secondary",
        className="mb-3",
    )


def make_gene_cluster_table(df):
    table_cols = [
        "Gene", "GeneName", "Cluster_DTW", "Silhouette_DTW", "Cluster_cosine", "Silhouette_cosine",
        "Cluster", "Silhouette", "Active_Clustering_Method", "Custom_Clustering_Mode", "Selected_Times", "Selected_Spaces",
    ]
    existing_cols = [c for c in table_cols if c in df.columns]
    out = df[existing_cols].copy()

    for col in ["Silhouette_DTW", "Silhouette_cosine", "Silhouette"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").round(4)

    return html.Div(
        [
            gene_cluster_table_explanation(),
            dash_table.DataTable(
                data=out.to_dict("records"),
                columns=[{"name": c, "id": c} for c in out.columns],
                page_size=15,
                filter_action="native",
                sort_action="native",
                style_table={"overflowX": "auto", "maxHeight": "650px", "overflowY": "auto"},
                style_cell={
                    "textAlign": "left",
                    "fontFamily": "Arial",
                    "fontSize": "13px",
                    "padding": "6px",
                    "minWidth": "110px",
                    "whiteSpace": "normal",
                },
                style_header={"fontWeight": "bold", "backgroundColor": "#f1f3f5"},
            ),
        ]
    )


def render_result_content(
    active_tab,
    gene_clusters,
    all_tb,
    selected_clusters,
    visualization_mode,
    visualization_time,
    visualization_space,
    max_curves,
    plot_options,
    individual_curve_style,
    curve_fit_method,
    loess_span,
    display_window_mode="full",
    selected_times=None,
    selected_spaces=None,
):
    plot_options = plot_options or []
    selected_clusters = selected_clusters or []

    gene_clusters_filtered, all_tb_filtered = filter_to_selected_clusters(gene_clusters, all_tb, selected_clusters)

    if active_tab == "gene_table":
        return make_gene_cluster_table(gene_clusters_filtered)

    display_times = [t for t in (selected_times or TIME_LEVELS) if t in TIME_LEVELS]
    display_spaces = [s for s in (selected_spaces or SPACE_LEVELS) if s in SPACE_LEVELS]
    if display_window_mode != "selected":
        display_times = TIME_LEVELS.copy()
        display_spaces = SPACE_LEVELS.copy()

    if display_window_mode == "selected":
        window_note = dbc.Alert(
            [
                html.Strong("Plot display window: selected clustering window. "),
                "By default, Module 2 plots only the time and GI-tract location window used for customized clustering. ",
                "This keeps the displayed profiles consistent with the features used to define similarity and assign clusters.",
            ],
            color="secondary",
            className="mb-2",
        )
    else:
        window_note = dbc.Alert(
            [
                html.Strong("Plot display window: full dataset. "),
                "The figure shows the full time and GI-tract location window, including values that were not used for customized clustering. ",
                "Use this for visual context only: unselected timepoints or locations may not share the similarity structure used for the current cluster assignment.",
            ],
            color="warning",
            className="mb-2",
        )

    show_median = "median" in plot_options
    fixed_y_axis = "fixed_y" in plot_options
    loess_span = safe_float(loess_span, default=0.8, minimum=0.1, maximum=1.0)

    if visualization_mode == "global_3d":
        fig = make_global_3d_cluster_figure(
            all_tb_filtered,
            display_times=display_times,
            display_spaces=display_spaces,
        )
    else:
        fig = make_faceted_profile_figure(
            all_tb=all_tb_filtered,
            visualization_mode=visualization_mode,
            visualization_time=visualization_time,
            visualization_space=visualization_space,
            max_genes_per_cluster=max_curves,
            show_median=show_median,
            curve_fit_method=curve_fit_method,
            loess_span=loess_span,
            fixed_y_axis=fixed_y_axis,
            individual_curve_style=individual_curve_style,
            display_times=display_times,
            display_spaces=display_spaces,
        )

    return html.Div([window_note, dcc.Graph(figure=fig, style={"width": "100%"})])


# ============================================================
# 6. Layout pieces
# ============================================================

def visualization_controls(prefix):
    return html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Label("Visualization type"),
                            dcc.Dropdown(
                                id=f"{prefix}-visualization-mode",
                                options=[
                                    {"label": "Spatial profiles at selected time point", "value": "spatial_profile"},
                                    {"label": "Temporal profiles at selected space point", "value": "temporal_profile"},
                                    {"label": "3D cluster median profiles", "value": "global_3d"},
                                ],
                                value=DEFAULT_VISUALIZATION_MODE,
                                clearable=False,
                            ),
                        ],
                        md=3,
                    ),
                    dbc.Col(
                        [
                            html.Label("Time point for spatial-profile visualization"),
                            dcc.Dropdown(
                                id=f"{prefix}-visualization-time",
                                options=[{"label": t, "value": t} for t in TIME_LEVELS],
                                value="12h",
                                clearable=False,
                            ),
                        ],
                        md=3,
                    ),
                    dbc.Col(
                        [
                            html.Label("Space point for temporal-profile visualization"),
                            dcc.Dropdown(
                                id=f"{prefix}-visualization-space",
                                options=[{"label": s, "value": s} for s in SPACE_LEVELS],
                                value="SI8",
                                clearable=False,
                            ),
                        ],
                        md=3,
                    ),
                    dbc.Col(
                        [
                            html.Label("Max gene curves per cluster"),
                            dbc.Input(
                                id=f"{prefix}-max-curves",
                                type="number",
                                min=0,
                                max=5000,
                                step=10,
                                value=250,
                            ),
                        ],
                        md=3,
                    ),
                ],
                className="mb-3",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Label("Plot options"),
                            dcc.Checklist(
                                id=f"{prefix}-plot-options",
                                options=[
                                    {"label": "Show cluster median curve", "value": "median"},
                                    {"label": "Use fixed Y-axis across clusters", "value": "fixed_y"},
                                ],
                                value=["median"],
                                inline=False,
                            ),
                        ],
                        md=3,
                    ),
                    dbc.Col(
                        [
                            html.Label("Individual gene curve style"),
                            dcc.Dropdown(
                                id=f"{prefix}-individual-curve-style",
                                options=[
                                    {"label": "Grey individual curves", "value": "grey"},
                                    {"label": "Colored individual curves", "value": "colored"},
                                ],
                                value="grey",
                                clearable=False,
                            ),
                        ],
                        md=3,
                    ),
                    dbc.Col(
                        [
                            html.Label("Curve fitting method"),
                            dcc.Dropdown(
                                id=f"{prefix}-curve-fit-method",
                                options=[
                                    {"label": "Raw connected line", "value": "none"},
                                    {"label": "LOESS", "value": "loess"},
                                    {"label": "Polynomial degree 2", "value": "poly2"},
                                    {"label": "Polynomial degree 3", "value": "poly3"},
                                ],
                                value="none",
                                clearable=False,
                            ),
                        ],
                        md=3,
                    ),
                    dbc.Col(
                        [
                            html.Label("LOESS span"),
                            dbc.Input(
                                id=f"{prefix}-loess-span",
                                type="number",
                                min=0.1,
                                max=1.0,
                                step=0.05,
                                value=0.8,
                            ),
                            html.Small("Used only when LOESS is selected. Default = 0.8.", className="text-muted"),
                        ],
                        id=f"{prefix}-loess-span-wrapper",
                        md=3,
                        style={"display": "none"},
                    ),
                ],
                className="mb-3",
            ),
        ]
    )


# ============================================================
# 7. Layout
# ============================================================

layout = dbc.Container(
    [
        html.H2("Clustering", className="page-title"),
        html.P(
            "Cluster genes based on their spatial, temporal, or global spatial-temporal in vivo fitness profiles. "
            "The page is split into a fast recommended module using precomputed 3D global DTW clustering and a slower "
            "customized clustering module.",
            className="lead",
        ),
        dbc.Alert(
            [
                html.Strong("Method overview: "),
                html.Span(
                    "The clustering logic follows the uploaded R workflow: each gene is converted into a fitness trajectory/vector, "
                    "DTW compares trajectory shape with a small warping window, cosine compares vector direction, and silhouette "
                    "scores quantify cluster assignment quality. Precomputed results are recommended for routine exploration; "
                    "customized clustering recomputes distances on the server and can take time."
                ),
            ],
            color="secondary",
            className="mb-4",
        ),

        # =====================================================
        # MODULE 1
        # =====================================================
        html.H3("1. Recommended precomputed clustering"),
        dbc.Alert(
            [
                html.Strong("Recommended default: "),
                html.Span(
                    "This module uses the precomputed 3D global DTW clustering result. Clustering mode is fixed to "
                    "3D global clustering, number of clusters is fixed to 20, and the active clustering method is fixed to DTW. "
                    "No clustering analysis is run on the server here, so this module should load quickly and is recommended "
                    "for routine database browsing."
                ),
            ],
            color="info",
            className="mb-3",
        ),
        dbc.Row(
            [
                dbc.Col(dbc.Card(dbc.CardBody([html.Strong("Clustering mode"), html.Br(), "3D global clustering"])), md=4),
                dbc.Col(dbc.Card(dbc.CardBody([html.Strong("Number of clusters"), html.Br(), "20"])), md=4),
                dbc.Col(dbc.Card(dbc.CardBody([html.Strong("Active clustering method"), html.Br(), "DTW distance"])), md=4),
            ],
            className="mb-3",
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Label("Show selected clusters only"),
                        dcc.Dropdown(
                            id="module1-cluster-filter",
                            options=[],
                            value=[],
                            multi=True,
                            placeholder="Leave empty to show all clusters",
                        ),
                    ],
                    md=6,
                ),
            ],
            className="mb-3",
        ),
        visualization_controls("module1"),
        dcc.Tabs(
            id="module1-tabs",
            value="plot",
            children=[
                dcc.Tab(label="Cluster Plot", value="plot"),
                dcc.Tab(label="Gene Cluster Table", value="gene_table"),
            ],
        ),
        html.Br(),
        dcc.Loading(
            type="circle",
            children=[html.Div(id="module1-summary"), html.Div(id="module1-content")],
        ),
        dbc.Button(
            "Download precomputed gene cluster table",
            id="module1-download-button",
            color="secondary",
            size="sm",
            className="mt-2 mb-5",
        ),
        dcc.Download(id="module1-download"),

        html.Hr(),

        # =====================================================
        # MODULE 2
        # =====================================================
        html.H3("2. Customized clustering analysis"),
        dbc.Alert(
            [
                html.Strong("Custom analysis: "),
                html.Span(
                    "This module recomputes clustering from the raw fitness matrix on the server. It supports 3D time × space "
                    "clustering, 2D spatial clustering, and 2D temporal clustering within a selected time/space window. "
                    "Because pairwise distance calculation is expensive, especially for DTW, it may take noticeable time. "
                    "Use the progress/status bar and avoid repeatedly changing options during a running analysis."
                ),
            ],
            color="warning",
            className="mb-3",
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Label("Customized clustering mode"),
                        dcc.Dropdown(
                            id="module2-cluster-mode",
                            options=[
                                {"label": "3D global clustering using selected time × space window", "value": "global"},
                                {"label": "2D spatial clustering across selected GI-tract locations", "value": "spatial"},
                                {"label": "2D temporal clustering across selected time points", "value": "temporal"},
                            ],
                            value="global",
                            clearable=False,
                        ),
                    ],
                    md=4,
                ),
                dbc.Col(
                    [
                        html.Label("Number of clusters"),
                        dbc.Input(id="module2-cluster-k", type="number", min=2, max=100, step=1, value=20),
                    ],
                    md=2,
                ),
                dbc.Col(
                    [
                        html.Label("Clustering method"),
                        dcc.Dropdown(
                            id="module2-cluster-method",
                            options=[
                                {"label": "DTW distance-based clustering", "value": "dtw"},
                                {"label": "Cosine distance-based clustering", "value": "cosine"},
                            ],
                            value="dtw",
                            clearable=False,
                        ),
                    ],
                    md=3,
                ),
                dbc.Col(
                    [
                        html.Label("Preprocessing"),
                        dcc.Checklist(
                            id="module2-preprocess-options",
                            options=[{"label": "Z-score scale each gene before clustering", "value": "scale"}],
                            value=[],
                            inline=False,
                        ),
                    ],
                    md=3,
                ),
            ],
            className="mb-3",
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Label("Start time"),
                        dcc.Dropdown(id="module2-start-time", options=[{"label": t, "value": t} for t in TIME_LEVELS], value=TIME_LEVELS[0], clearable=False),
                    ],
                    md=3,
                ),
                dbc.Col(
                    [
                        html.Label("End time"),
                        dcc.Dropdown(id="module2-end-time", options=[{"label": t, "value": t} for t in TIME_LEVELS], value=TIME_LEVELS[-1], clearable=False),
                    ],
                    md=3,
                ),
                dbc.Col(
                    [
                        html.Label("Start space"),
                        dcc.Dropdown(id="module2-start-space", options=[{"label": s, "value": s} for s in SPACE_LEVELS], value=SPACE_LEVELS[0], clearable=False),
                    ],
                    md=3,
                ),
                dbc.Col(
                    [
                        html.Label("End space"),
                        dcc.Dropdown(id="module2-end-space", options=[{"label": s, "value": s} for s in SPACE_LEVELS], value=SPACE_LEVELS[-1], clearable=False),
                    ],
                    md=3,
                ),
            ],
            className="mb-3",
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Button("Run customized clustering", id="module2-run-button", color="primary", className="me-2"),
                        html.Small(
                            "DTW on all genes can be slow because the app computes pairwise trajectory distances.",
                            className="text-muted",
                        ),
                    ],
                    md=5,
                ),
                dbc.Col(
                    [
                        dbc.Progress(
                            id="module2-progress",
                            value=0,
                            label="Not started",
                            color="secondary",
                            striped=True,
                            animated=True,
                            className="mb-1",
                        ),
                        html.Small(
                            "After clicking Run, the progress bar shows that customized clustering is running. "
                            "A completion message appears when the analysis finishes.",
                            className="text-muted",
                        ),
                    ],
                    md=7,
                ),
            ],
            className="mb-3",
        ),
        dcc.Store(id="module2-run-status-store", data={"status": "not_started", "run_id": 0}),
        dcc.Interval(id="module2-progress-interval", interval=700, n_intervals=0, disabled=True),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Label("Module 2 plot display window"),
                        dcc.RadioItems(
                            id="module2-plot-window-mode",
                            options=[
                                {
                                    "label": "Show only the selected clustering window (recommended)",
                                    "value": "selected",
                                },
                                {
                                    "label": "Show full time and GI-tract location window",
                                    "value": "full",
                                },
                            ],
                            value="selected",
                            inline=False,
                            inputStyle={"marginRight": "6px"},
                            labelStyle={"display": "block", "marginBottom": "4px"},
                        ),
                        html.Small(
                            "Default: the cluster plot shows only the time/space window used for customized clustering. "
                            "Showing the full window can be useful for context, but unselected windows were not used to define similarity and may not follow the current cluster pattern.",
                            className="text-muted",
                        ),
                    ],
                    md=8,
                ),
            ],
            className="mb-3",
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Label("Show selected clusters only"),
                        dcc.Dropdown(
                            id="module2-cluster-filter",
                            options=[],
                            value=[],
                            multi=True,
                            placeholder="Run custom clustering first; leave empty to show all clusters",
                        ),
                    ],
                    md=6,
                ),
            ],
            className="mb-3",
        ),
        visualization_controls("module2"),
        dcc.Tabs(
            id="module2-tabs",
            value="plot",
            children=[
                dcc.Tab(label="Cluster Plot", value="plot"),
                dcc.Tab(label="Gene Cluster Table", value="gene_table"),
            ],
        ),
        html.Br(),
        dcc.Loading(
            type="circle",
            children=[html.Div(id="module2-summary"), html.Div(id="module2-content")],
        ),
        dbc.Button(
            "Download customized gene cluster table",
            id="module2-download-button",
            color="secondary",
            size="sm",
            className="mt-2 mb-5",
        ),
        dcc.Download(id="module2-download"),
    ],
    fluid=True,
)


# ============================================================
# 8. LOESS control visibility callbacks
# ============================================================

@dash.callback(
    Output("module1-loess-span-wrapper", "style"),
    Input("module1-curve-fit-method", "value"),
)
def toggle_module1_loess_span(curve_fit_method):
    if curve_fit_method == "loess":
        return {"display": "block"}
    return {"display": "none"}


@dash.callback(
    Output("module2-loess-span-wrapper", "style"),
    Input("module2-curve-fit-method", "value"),
)
def toggle_module2_loess_span(curve_fit_method):
    if curve_fit_method == "loess":
        return {"display": "block"}
    return {"display": "none"}


# ============================================================
# 9. Module 1 callbacks
# ============================================================

@dash.callback(
    Output("module1-cluster-filter", "options"),
    Output("module1-cluster-filter", "value"),
    Input("module1-tabs", "value"),
    State("module1-cluster-filter", "value"),
)
def initialize_module1_cluster_filter(_tab, current_value):
    try:
        gene_clusters, _all_tb = get_precomputed_result_for_module1()
        options = make_cluster_options(gene_clusters)
        allowed = {opt["value"] for opt in options}
        current_value = current_value or []
        kept = [int(v) for v in current_value if int(v) in allowed]
        return options, kept
    except Exception:
        return [], []


@dash.callback(
    Output("module1-summary", "children"),
    Output("module1-content", "children"),
    Input("module1-tabs", "value"),
    Input("module1-cluster-filter", "value"),
    Input("module1-visualization-mode", "value"),
    Input("module1-visualization-time", "value"),
    Input("module1-visualization-space", "value"),
    Input("module1-max-curves", "value"),
    Input("module1-plot-options", "value"),
    Input("module1-individual-curve-style", "value"),
    Input("module1-curve-fit-method", "value"),
    Input("module1-loess-span", "value"),
)
def update_module1(
    active_tab,
    selected_clusters,
    visualization_mode,
    visualization_time,
    visualization_space,
    max_curves,
    plot_options,
    individual_curve_style,
    curve_fit_method,
    loess_span,
):
    try:
        gene_clusters, all_tb = get_precomputed_result_for_module1()
        selected_times = TIME_LEVELS
        selected_spaces = SPACE_LEVELS
        selected_clusters = selected_clusters or []

        gene_clusters_filtered, _ = filter_to_selected_clusters(gene_clusters, all_tb, selected_clusters)
        summary = summarize_result(
            gene_clusters=gene_clusters_filtered,
            mode="global",
            k=20,
            method_label="DTW distance",
            source_text="Precomputed 3D global DTW clustering table; no server-side clustering was run",
            selected_times=selected_times,
            selected_spaces=selected_spaces,
            selected_clusters=selected_clusters,
        )
        content = render_result_content(
            active_tab=active_tab,
            gene_clusters=gene_clusters,
            all_tb=all_tb,
            selected_clusters=selected_clusters,
            visualization_mode=visualization_mode,
            visualization_time=visualization_time,
            visualization_space=visualization_space,
            max_curves=max_curves,
            plot_options=plot_options,
            individual_curve_style=individual_curve_style,
            curve_fit_method=curve_fit_method,
            loess_span=loess_span,
        )
        return summary, content
    except Exception as e:
        return dbc.Alert(str(e), color="danger"), ""


@dash.callback(
    Output("module1-download", "data"),
    Input("module1-download-button", "n_clicks"),
    prevent_initial_call=True,
)
def download_module1_table(n_clicks):
    if not n_clicks:
        return no_update
    gene_clusters, _all_tb = get_precomputed_result_for_module1()
    out = gene_clusters.copy()
    return dcc.send_data_frame(out.to_csv, "precomputed_3D_global_DTW_k20_gene_clusters.csv", index=False)


# ============================================================
# 9. Module 2 callbacks
# ============================================================

@dash.callback(
    Output("module2-cluster-filter", "options"),
    Output("module2-cluster-filter", "value"),
    Input("module2-run-button", "n_clicks"),
    State("module2-cluster-mode", "value"),
    State("module2-cluster-k", "value"),
    State("module2-cluster-method", "value"),
    State("module2-preprocess-options", "value"),
    State("module2-start-time", "value"),
    State("module2-end-time", "value"),
    State("module2-start-space", "value"),
    State("module2-end-space", "value"),
)
def update_module2_cluster_filter(
    n_clicks,
    mode,
    k,
    method,
    preprocess_options,
    start_time,
    end_time,
    start_space,
    end_space,
):
    if not n_clicks:
        return [], []

    try:
        scale_per_gene = "scale" in (preprocess_options or [])
        gene_clusters, _all_tb, _feature_labels, _selected_times, _selected_spaces = compute_custom_clustering_cached(
            mode=mode,
            k=safe_int(k, default=20, minimum=2, maximum=100),
            method=method,
            scale_per_gene=scale_per_gene,
            start_time=start_time,
            end_time=end_time,
            start_space=start_space,
            end_space=end_space,
        )
        return make_cluster_options(gene_clusters), []
    except Exception:
        return [], []


@dash.callback(
    Output("module2-progress", "value"),
    Output("module2-progress", "label"),
    Output("module2-progress", "color"),
    Output("module2-progress-interval", "disabled"),
    Input("module2-run-button", "n_clicks"),
    Input("module2-progress-interval", "n_intervals"),
    Input("module2-run-status-store", "data"),
    State("module2-progress", "value"),
)
def update_module2_progress(n_clicks, n_intervals, status_data, current_value):
    status_data = status_data or {"status": "not_started"}
    status = status_data.get("status", "not_started")

    if not n_clicks:
        return 0, "Not started", "secondary", True

    triggered = dash.callback_context.triggered[0]["prop_id"].split(".")[0] if dash.callback_context.triggered else ""

    if triggered == "module2-run-button":
        return 8, "Running customized clustering...", "info", False

    if status == "finished":
        return 100, "Finished", "success", True

    if status == "error":
        return 100, "Error", "danger", True

    current_value = safe_int(current_value, default=8, minimum=0, maximum=90)
    next_value = min(90, current_value + 6)
    return next_value, f"Running customized clustering... {next_value}%", "info", False


@dash.callback(
    Output("module2-summary", "children"),
    Output("module2-content", "children"),
    Output("module2-run-status-store", "data"),
    Input("module2-run-button", "n_clicks"),
    Input("module2-tabs", "value"),
    Input("module2-cluster-filter", "value"),
    Input("module2-visualization-mode", "value"),
    Input("module2-visualization-time", "value"),
    Input("module2-visualization-space", "value"),
    Input("module2-max-curves", "value"),
    Input("module2-plot-options", "value"),
    Input("module2-individual-curve-style", "value"),
    Input("module2-curve-fit-method", "value"),
    Input("module2-loess-span", "value"),
    Input("module2-plot-window-mode", "value"),
    State("module2-cluster-mode", "value"),
    State("module2-cluster-k", "value"),
    State("module2-cluster-method", "value"),
    State("module2-preprocess-options", "value"),
    State("module2-start-time", "value"),
    State("module2-end-time", "value"),
    State("module2-start-space", "value"),
    State("module2-end-space", "value"),
)
def update_module2(
    n_clicks,
    active_tab,
    selected_clusters,
    visualization_mode,
    visualization_time,
    visualization_space,
    max_curves,
    plot_options,
    individual_curve_style,
    curve_fit_method,
    loess_span,
    plot_window_mode,
    mode,
    k,
    method,
    preprocess_options,
    start_time,
    end_time,
    start_space,
    end_space,
):
    if not n_clicks:
        return (
            dbc.Alert("Set customized clustering options and click Run customized clustering.", color="secondary"),
            "",
            {"status": "not_started", "run_id": 0},
        )

    try:
        scale_per_gene = "scale" in (preprocess_options or [])
        k_safe = safe_int(k, default=20, minimum=2, maximum=100)
        gene_clusters, all_tb, feature_labels, selected_times, selected_spaces = compute_custom_clustering_cached(
            mode=mode,
            k=k_safe,
            method=method,
            scale_per_gene=scale_per_gene,
            start_time=start_time,
            end_time=end_time,
            start_space=start_space,
            end_space=end_space,
        )

        method_label = "DTW distance" if method == "dtw" else "Cosine distance"
        selected_clusters = selected_clusters or []
        gene_clusters_filtered, _ = filter_to_selected_clusters(gene_clusters, all_tb, selected_clusters)
        summary = html.Div(
            [
                summarize_result(
                    gene_clusters=gene_clusters_filtered,
                    mode=mode,
                    k=k_safe,
                    method_label=method_label,
                    source_text="Customized clustering computed from the raw fitness matrix",
                    selected_times=selected_times,
                    selected_spaces=selected_spaces,
                    selected_clusters=selected_clusters,
                ),
                dbc.Alert(
                    [
                        html.Strong("Custom feature matrix: "),
                        f"{len(feature_labels)} features per gene. ",
                        "For 3D mode, features are time × space values. For 2D spatial mode, selected timepoints are averaged for each space. "
                        "For 2D temporal mode, selected spaces are averaged for each timepoint.",
                    ],
                    color="secondary",
                    className="mb-3",
                ),
                dbc.Alert(
                    [
                        html.Strong("Customized clustering finished successfully. "),
                        f"Computed {gene_clusters['Gene'].nunique()} genes into {gene_clusters['Cluster'].nunique()} clusters using {method_label}.",
                    ],
                    color="success",
                    className="mb-3",
                ),
            ]
        )
        content = render_result_content(
            active_tab=active_tab,
            gene_clusters=gene_clusters,
            all_tb=all_tb,
            selected_clusters=selected_clusters,
            visualization_mode=visualization_mode,
            visualization_time=visualization_time,
            visualization_space=visualization_space,
            max_curves=max_curves,
            plot_options=plot_options,
            individual_curve_style=individual_curve_style,
            curve_fit_method=curve_fit_method,
            loess_span=loess_span,
            display_window_mode=plot_window_mode,
            selected_times=selected_times,
            selected_spaces=selected_spaces,
        )

        return summary, content, {"status": "finished", "run_id": int(n_clicks or 0)}

    except Exception as e:
        return dbc.Alert(str(e), color="danger"), "", {"status": "error", "run_id": int(n_clicks or 0), "message": str(e)}


@dash.callback(
    Output("module2-download", "data"),
    Input("module2-download-button", "n_clicks"),
    State("module2-cluster-mode", "value"),
    State("module2-cluster-k", "value"),
    State("module2-cluster-method", "value"),
    State("module2-preprocess-options", "value"),
    State("module2-start-time", "value"),
    State("module2-end-time", "value"),
    State("module2-start-space", "value"),
    State("module2-end-space", "value"),
    prevent_initial_call=True,
)
def download_module2_table(
    n_clicks,
    mode,
    k,
    method,
    preprocess_options,
    start_time,
    end_time,
    start_space,
    end_space,
):
    if not n_clicks:
        return no_update

    scale_per_gene = "scale" in (preprocess_options or [])
    k_safe = safe_int(k, default=20, minimum=2, maximum=100)
    gene_clusters, _all_tb, _feature_labels, selected_times, selected_spaces = compute_custom_clustering_cached(
        mode=mode,
        k=k_safe,
        method=method,
        scale_per_gene=scale_per_gene,
        start_time=start_time,
        end_time=end_time,
        start_space=start_space,
        end_space=end_space,
    )

    safe_times = "-".join(selected_times)
    safe_spaces = "-".join(selected_spaces)
    scale_label = "scaled" if scale_per_gene else "unscaled"
    filename = f"custom_clustering_{mode}_{method}_k{k_safe}_{scale_label}_time-{safe_times}_space-{safe_spaces}.csv"
    return dcc.send_data_frame(gene_clusters.to_csv, filename, index=False)
