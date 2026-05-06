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
    "Spatial_temporal_MultiSCAST_FC_final_capping.csv"
)

ANNOTATION_FILE = os.path.join(
    DATA_DIR,
    "annotation",
    "new_annotations_with_uniprot_names.csv"
)

PRECOMPUTED_CLUSTER_FILE = os.path.join(
    DATA_DIR,
    "clustering",
    "invivo_DTW_cosine_clustering_3Dlevel.csv"
)

SPACE_LEVELS = [
    "st", "SI1", "SI2", "SI3", "SI4", "SI5",
    "SI6", "SI7", "SI8", "SI9", "ce", "co"
]

TIME_LEVELS = ["1h", "3h", "6h", "12h", "24h"]

TIME_NUM_MAP = {
    "1h": 1,
    "3h": 3,
    "6h": 6,
    "12h": 12,
    "24h": 24,
}

SPACE_NUM_MAP = {s: i + 1 for i, s in enumerate(SPACE_LEVELS)}

DEFAULT_K = 20
DEFAULT_CLUSTERING_MODE = "global"
DEFAULT_CLUSTERING_METHOD = "dtw"

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

    required_cols = {"Gene", "Time", "Space", "logFC"}
    missing = required_cols - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns in raw data: {missing}")

    df = df[["Gene", "Time", "Space", "logFC"]].copy()

    df["Gene"] = df["Gene"].astype(str)
    df["Time"] = df["Time"].astype(str).str.strip()
    df["Space"] = df["Space"].astype(str).str.strip()
    df["logFC"] = pd.to_numeric(df["logFC"], errors="coerce")

    df = df[df["Time"].isin(TIME_LEVELS)]
    df = df[df["Space"].isin(SPACE_LEVELS)]

    df["Time"] = pd.Categorical(df["Time"], categories=TIME_LEVELS, ordered=True)
    df["Space"] = pd.Categorical(df["Space"], categories=SPACE_LEVELS, ordered=True)
    df["TimeNum"] = df["Time"].astype(str).map(TIME_NUM_MAP)
    df["SpaceNum"] = df["Space"].astype(str).map(SPACE_NUM_MAP)

    return df


@lru_cache(maxsize=1)
def load_precomputed_cluster_data():
    """
    Load precomputed 3D global clustering results.

    Expected useful columns:
      Gene, Time, Space, logFC, Cluster_DTW, Cluster_cosine

    Optional useful columns:
      Silhouette_DTW, Silhouette_cosine
    """
    if not os.path.exists(PRECOMPUTED_CLUSTER_FILE):
        return None

    df = pd.read_csv(PRECOMPUTED_CLUSTER_FILE)

    required_base = {"Gene", "Time", "Space", "logFC"}
    missing = required_base - set(df.columns)

    if missing:
        raise ValueError(
            f"Precomputed clustering file is missing required columns: {missing}. "
            f"File path: {PRECOMPUTED_CLUSTER_FILE}"
        )

    df = df.copy()
    df["Gene"] = df["Gene"].astype(str)
    df["Time"] = df["Time"].astype(str).str.strip()
    df["Space"] = df["Space"].astype(str).str.strip()
    df["logFC"] = pd.to_numeric(df["logFC"], errors="coerce")

    df = df[df["Time"].isin(TIME_LEVELS)]
    df = df[df["Space"].isin(SPACE_LEVELS)]

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
    ann["locus_ID"] = ann["locus_ID"].astype(str)
    ann["gene_name"] = ann["gene_name"].fillna("").astype(str)

    ann["GeneName"] = np.where(
        ann["gene_name"].str.strip() != "",
        ann["gene_name"],
        ann["locus_ID"]
    )

    ann = (
        ann.rename(columns={"locus_ID": "Gene"})
        [["Gene", "GeneName"]]
        .drop_duplicates()
    )

    return ann


def add_annotation(df):
    """
    Add GeneName from annotation table.

    Robust when the input dataframe already contains
    GeneName, gene_name, GeneName_x, or GeneName_y columns.
    """
    ann = load_annotation().copy()
    out = df.copy()

    if "Gene" not in out.columns:
        raise ValueError("Input dataframe must contain a 'Gene' column for annotation merge.")

    annotation_like_cols = [
        "GeneName",
        "gene_name",
        "GeneName_x",
        "GeneName_y",
        "gene_name_x",
        "gene_name_y",
    ]

    drop_cols = [c for c in annotation_like_cols if c in out.columns]
    if drop_cols:
        out = out.drop(columns=drop_cols)

    out["Gene"] = out["Gene"].astype(str)
    ann["Gene"] = ann["Gene"].astype(str)

    out = out.merge(ann[["Gene", "GeneName"]], on="Gene", how="left")
    out["GeneName"] = out["GeneName"].fillna(out["Gene"])

    return out


# ============================================================
# 2. Column helpers
# ============================================================

def find_first_existing_column(df, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    return None


def get_precomputed_cluster_column(clustering_method):
    pre = load_precomputed_cluster_data()

    if pre is None:
        return None

    if clustering_method == "dtw":
        candidates = ["Cluster_DTW", "cluster_DTW", "Cluster_dtw", "cluster_dtw"]
    else:
        candidates = ["Cluster_cosine", "cluster_cosine", "Cluster_Cosine", "cluster_Cosine"]

    return find_first_existing_column(pre, candidates)


def get_precomputed_dtw_cluster_column():
    pre = load_precomputed_cluster_data()
    if pre is None:
        return None

    return find_first_existing_column(
        pre,
        ["Cluster_DTW", "cluster_DTW", "Cluster_dtw", "cluster_dtw"]
    )


def get_precomputed_cosine_cluster_column():
    pre = load_precomputed_cluster_data()
    if pre is None:
        return None

    return find_first_existing_column(
        pre,
        ["Cluster_cosine", "cluster_cosine", "Cluster_Cosine", "cluster_Cosine"]
    )


def get_precomputed_dtw_silhouette_column():
    pre = load_precomputed_cluster_data()
    if pre is None:
        return None

    return find_first_existing_column(
        pre,
        [
            "Silhouette_DTW", "silhouette_DTW", "Silhouette_dtw", "silhouette_dtw",
            "sil_DTW", "sil_dtw", "Sil_DTW", "Sil_dtw"
        ]
    )


def get_precomputed_cosine_silhouette_column():
    pre = load_precomputed_cluster_data()
    if pre is None:
        return None

    return find_first_existing_column(
        pre,
        [
            "Silhouette_cosine", "silhouette_cosine", "Silhouette_Cosine", "silhouette_Cosine",
            "sil_cosine", "Sil_cosine", "sil_Cosine", "Sil_Cosine"
        ]
    )


# ============================================================
# 3. Feature matrix construction
# ============================================================

def zscore_rows(X):
    X = X.astype(float)
    mean = np.nanmean(X, axis=1, keepdims=True)
    sd = np.nanstd(X, axis=1, keepdims=True)

    sd[~np.isfinite(sd)] = 1.0
    sd[sd == 0] = 1.0

    return (X - mean) / sd


def build_feature_matrix(mode, selected_time, selected_space, scale_per_gene):
    """
    mode:
      spatial  = each gene represented by 12 spatial values at one selected time
      temporal = each gene represented by 5 temporal values at one selected space
      global   = each gene represented by 5 x 12 = 60 spatial-temporal values
    """
    df = load_raw_data()
    genes = sorted(df["Gene"].unique())

    if mode == "spatial":
        sub = df[df["Time"].astype(str) == selected_time].copy()

        complete_index = pd.MultiIndex.from_product(
            [genes, SPACE_LEVELS],
            names=["Gene", "Space"]
        )

        feature_df = (
            sub[["Gene", "Space", "logFC"]]
            .set_index(["Gene", "Space"])
            .reindex(complete_index)
            .reset_index()
        )

        feature_df["logFC"] = feature_df["logFC"].fillna(0)

        Xdf = (
            feature_df
            .pivot(index="Gene", columns="Space", values="logFC")
            .reindex(index=genes, columns=SPACE_LEVELS)
            .fillna(0)
        )

        feature_labels = SPACE_LEVELS

    elif mode == "temporal":
        sub = df[df["Space"].astype(str) == selected_space].copy()

        complete_index = pd.MultiIndex.from_product(
            [genes, TIME_LEVELS],
            names=["Gene", "Time"]
        )

        feature_df = (
            sub[["Gene", "Time", "logFC"]]
            .set_index(["Gene", "Time"])
            .reindex(complete_index)
            .reset_index()
        )

        feature_df["logFC"] = feature_df["logFC"].fillna(0)

        Xdf = (
            feature_df
            .pivot(index="Gene", columns="Time", values="logFC")
            .reindex(index=genes, columns=TIME_LEVELS)
            .fillna(0)
        )

        feature_labels = TIME_LEVELS

    else:
        complete_index = pd.MultiIndex.from_product(
            [genes, TIME_LEVELS, SPACE_LEVELS],
            names=["Gene", "Time", "Space"]
        )

        feature_df = (
            df[["Gene", "Time", "Space", "logFC"]]
            .set_index(["Gene", "Time", "Space"])
            .reindex(complete_index)
            .reset_index()
        )

        feature_df["logFC"] = feature_df["logFC"].fillna(0)

        feature_df["Feature"] = (
            feature_df["Time"].astype(str) + "_" + feature_df["Space"].astype(str)
        )

        feature_order = [
            f"{t}_{s}"
            for t in TIME_LEVELS
            for s in SPACE_LEVELS
        ]

        Xdf = (
            feature_df
            .pivot(index="Gene", columns="Feature", values="logFC")
            .reindex(index=genes, columns=feature_order)
            .fillna(0)
        )

        feature_labels = feature_order

    X = Xdf.values.astype(float)

    if scale_per_gene:
        X = zscore_rows(X)

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    return Xdf.index.tolist(), X, feature_labels


# ============================================================
# 4. Distance and clustering
# ============================================================

def cosine_distance_vectorized(X):
    Dvec = pdist(X, metric="cosine")
    Dvec = np.nan_to_num(Dvec, nan=0.0, posinf=0.0, neginf=0.0)
    return Dvec


def dtw_distance_1d(x, y, window=None):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    n = len(x)
    m = len(y)

    if window is None:
        window = max(n, m)
    else:
        window = max(int(window), abs(n - m))

    inf = np.inf
    dtw = np.full((n + 1, m + 1), inf)
    dtw[0, 0] = 0.0

    for i in range(1, n + 1):
        j_start = max(1, i - window)
        j_end = min(m, i + window)

        for j in range(j_start, j_end + 1):
            cost = (x[i - 1] - y[j - 1]) ** 2
            dtw[i, j] = cost + min(
                dtw[i - 1, j],
                dtw[i, j - 1],
                dtw[i - 1, j - 1]
            )

    return float(np.sqrt(dtw[n, m]))


def dtw_distance_vector(X, window_fraction=0.05):
    n = X.shape[0]
    length = X.shape[1]

    window = max(1, int(round(window_fraction * length)))

    out = []

    for i in range(n - 1):
        xi = X[i, :]

        for j in range(i + 1, n):
            yj = X[j, :]
            d = dtw_distance_1d(xi, yj, window=window)
            out.append(d)

    Dvec = np.array(out, dtype=float)
    Dvec = np.nan_to_num(Dvec, nan=0.0, posinf=0.0, neginf=0.0)

    return Dvec


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

        if np.any(same):
            a = np.mean(D[i, same])
        else:
            a = 0.0

        b_vals = []

        for c in unique_clusters:
            if c == labels[i]:
                continue

            other = labels == c

            if np.any(other):
                b_vals.append(np.mean(D[i, other]))

        b = min(b_vals) if b_vals else 0.0

        denom = max(a, b)

        if denom == 0:
            sil[i] = 0.0
        else:
            sil[i] = (b - a) / denom

    return sil


def cluster_from_distance_vector(Dvec, k):
    n = int((1 + np.sqrt(1 + 8 * len(Dvec))) / 2)

    if np.all(Dvec == 0):
        labels = np.ones(n, dtype=int)
        sil = np.zeros(n)
        return labels, sil

    Z = linkage(Dvec, method="average")
    labels = fcluster(Z, t=int(k), criterion="maxclust").astype(int)

    D = squareform(Dvec)
    sil = compute_silhouette_from_distance(D, labels)

    return labels, sil


def compute_method_labels(X, k, method):
    if method == "dtw":
        Dvec = dtw_distance_vector(X, window_fraction=0.05)
    else:
        Dvec = cosine_distance_vectorized(X)

    labels, sil = cluster_from_distance_vector(Dvec, k)
    return labels, sil


@lru_cache(maxsize=64)
def compute_dual_clustering_cached(
    mode,
    k,
    scale_per_gene,
    selected_time,
    selected_space,
):
    """
    Compute both DTW and cosine clustering for the same feature matrix.
    This allows the gene cluster table to always show both:
      Cluster_DTW, Cluster_cosine, Silhouette_DTW, Silhouette_cosine
    """
    genes, X, feature_labels = build_feature_matrix(
        mode=mode,
        selected_time=selected_time,
        selected_space=selected_space,
        scale_per_gene=scale_per_gene,
    )

    n = len(genes)

    if n == 0:
        raise ValueError("No genes found for clustering.")

    k = int(k)

    if k < 2:
        k = 2

    if k > n:
        k = n

    labels_dtw, sil_dtw = compute_method_labels(X, k, "dtw")
    labels_cosine, sil_cosine = compute_method_labels(X, k, "cosine")

    gene_clusters = pd.DataFrame(
        {
            "Gene": genes,
            "Cluster_DTW": labels_dtw,
            "Silhouette_DTW": sil_dtw,
            "Cluster_cosine": labels_cosine,
            "Silhouette_cosine": sil_cosine,
        }
    )

    gene_clusters = add_annotation(gene_clusters)

    raw = load_raw_data()
    all_tb = raw.merge(
        gene_clusters[
            [
                "Gene",
                "GeneName",
                "Cluster_DTW",
                "Silhouette_DTW",
                "Cluster_cosine",
                "Silhouette_cosine",
            ]
        ],
        on="Gene",
        how="left"
    )

    return gene_clusters, all_tb, feature_labels


# ============================================================
# 5. Precomputed clustering result
# ============================================================

def can_use_precomputed_result(mode, k, preprocess_options):
    """
    Use precomputed result only for default 3D-level clustering:
      global mode, k=20, no z-score recomputation.
    """
    preprocess_options = preprocess_options or []

    if "scale" in preprocess_options:
        return False

    if mode != "global":
        return False

    if int(k) != 20:
        return False

    if load_precomputed_cluster_data() is None:
        return False

    if get_precomputed_dtw_cluster_column() is None:
        return False

    if get_precomputed_cosine_cluster_column() is None:
        return False

    return True


def get_precomputed_result(clustering_method):
    pre = load_precomputed_cluster_data()

    if pre is None:
        raise FileNotFoundError(
            f"Precomputed clustering file not found: {PRECOMPUTED_CLUSTER_FILE}"
        )

    dtw_col = get_precomputed_dtw_cluster_column()
    cosine_col = get_precomputed_cosine_cluster_column()

    if dtw_col is None or cosine_col is None:
        raise ValueError(
            "Cannot find precomputed cluster columns. Expected Cluster_DTW and Cluster_cosine."
        )

    dtw_sil_col = get_precomputed_dtw_silhouette_column()
    cosine_sil_col = get_precomputed_cosine_silhouette_column()

    all_tb = pre.copy()

    all_tb["Cluster_DTW"] = pd.to_numeric(all_tb[dtw_col], errors="coerce")
    all_tb["Cluster_cosine"] = pd.to_numeric(all_tb[cosine_col], errors="coerce")

    if dtw_sil_col is not None:
        all_tb["Silhouette_DTW"] = pd.to_numeric(all_tb[dtw_sil_col], errors="coerce")
    else:
        all_tb["Silhouette_DTW"] = np.nan

    if cosine_sil_col is not None:
        all_tb["Silhouette_cosine"] = pd.to_numeric(all_tb[cosine_sil_col], errors="coerce")
    else:
        all_tb["Silhouette_cosine"] = np.nan

    all_tb = all_tb.dropna(subset=["Cluster_DTW", "Cluster_cosine"]).copy()
    all_tb["Cluster_DTW"] = all_tb["Cluster_DTW"].astype(int)
    all_tb["Cluster_cosine"] = all_tb["Cluster_cosine"].astype(int)

    all_tb = add_annotation(all_tb)

    gene_clusters = (
        all_tb[
            [
                "Gene",
                "GeneName",
                "Cluster_DTW",
                "Silhouette_DTW",
                "Cluster_cosine",
                "Silhouette_cosine",
            ]
        ]
        .drop_duplicates()
        .sort_values(["Cluster_DTW", "Cluster_cosine", "GeneName", "Gene"])
        .reset_index(drop=True)
    )

    feature_labels = [
        f"{t}_{s}"
        for t in TIME_LEVELS
        for s in SPACE_LEVELS
    ]

    return gene_clusters, all_tb, feature_labels


def add_active_cluster_columns(gene_clusters, all_tb, clustering_method):
    gene_clusters = gene_clusters.copy()
    all_tb = all_tb.copy()

    if clustering_method == "dtw":
        gene_clusters["Cluster"] = gene_clusters["Cluster_DTW"]
        gene_clusters["Silhouette"] = gene_clusters["Silhouette_DTW"]
        all_tb["Cluster"] = all_tb["Cluster_DTW"]
        all_tb["Silhouette"] = all_tb["Silhouette_DTW"]
        method_label = "DTW distance"
    else:
        gene_clusters["Cluster"] = gene_clusters["Cluster_cosine"]
        gene_clusters["Silhouette"] = gene_clusters["Silhouette_cosine"]
        all_tb["Cluster"] = all_tb["Cluster_cosine"]
        all_tb["Silhouette"] = all_tb["Silhouette_cosine"]
        method_label = "Cosine distance"

    gene_clusters["Active_Clustering_Method"] = method_label
    all_tb["Active_Clustering_Method"] = method_label

    return gene_clusters, all_tb


# ============================================================
# 6. Plot helpers
# ============================================================

def summarize_clustering(
    gene_clusters,
    mode,
    k,
    clustering_method,
    selected_time,
    selected_space,
    visualization_mode,
    visualization_time,
    visualization_space,
    fixed_y_axis,
    curve_fit_method,
    loess_span,
    individual_curve_style,
    show_median,
    used_precomputed,
    selected_clusters,
):
    n_genes = gene_clusters["Gene"].nunique()
    n_clusters = gene_clusters["Cluster"].nunique()

    if "Silhouette" in gene_clusters.columns:
        mean_sil = pd.to_numeric(gene_clusters["Silhouette"], errors="coerce").mean(skipna=True)
    else:
        mean_sil = np.nan

    if mode == "spatial":
        mode_text = f"2D spatial clustering at time point {selected_time}"
    elif mode == "temporal":
        mode_text = f"2D temporal clustering at spatial location {selected_space}"
    else:
        mode_text = "3D global clustering using all 5 x 12 time-space values"

    method_text = "DTW distance-based clustering" if clustering_method == "dtw" else "Cosine distance-based clustering"

    if visualization_mode == "spatial_profile":
        vis_text = f"Spatial profiles across 12 locations at {visualization_time}, faceted by cluster"
    elif visualization_mode == "temporal_profile":
        vis_text = f"Temporal profiles across 5 time points at {visualization_space}, faceted by cluster"
    else:
        vis_text = "3D cluster median spatial-temporal profiles"

    y_axis_text = (
        "Fixed Y-axis across clusters"
        if fixed_y_axis
        else "Flexible Y-axis for each cluster facet"
    )

    curve_text_map = {
        "none": "Raw connected line",
        "loess": f"LOESS smoothing, span = {loess_span}",
        "poly2": "Polynomial degree 2",
        "poly3": "Polynomial degree 3",
    }

    curve_text = curve_text_map.get(curve_fit_method, curve_fit_method)

    style_text = (
        "Colored individual curves"
        if individual_curve_style == "colored"
        else "Grey individual curves"
    )

    median_text = "shown" if show_median else "hidden"

    source_text = (
        "Precomputed 3D clustering table"
        if used_precomputed
        else "Clustering computed on demand from raw fitness matrix"
    )

    if selected_clusters:
        cluster_text = ", ".join([str(x) for x in selected_clusters])
    else:
        cluster_text = "All clusters"

    children = [
        html.Strong("Clustering summary"),
        html.Br(),
        f"Data source: {source_text}",
        html.Br(),
        f"Active clustering method: {method_text}",
        html.Br(),
        f"Clustering mode: {mode_text}",
        html.Br(),
        f"Visualization: {vis_text}",
        html.Br(),
        f"Displayed clusters: {cluster_text}",
        html.Br(),
        f"Curve fitting method: {curve_text}",
        html.Br(),
        f"Individual curve style: {style_text}",
        html.Br(),
        f"Cluster median curve: {median_text}",
        html.Br(),
        f"Y-axis mode: {y_axis_text}",
        html.Br(),
        f"Number of displayed genes: {n_genes}",
        html.Br(),
        f"Number of clusters requested: {k}",
        html.Br(),
        f"Number of displayed clusters: {n_clusters}",
    ]

    if np.isfinite(mean_sil):
        children.extend(
            [
                html.Br(),
                f"Mean silhouette score for active method: {mean_sil:.3f}",
            ]
        )
    else:
        children.extend(
            [
                html.Br(),
                "Mean silhouette score for active method: not available",
            ]
        )

    return dbc.Alert(
        children,
        color="info",
        className="mb-3"
    )


def fit_curve(x_num, y, fit_method="loess", span=0.8):
    """
    Fit or smooth a curve for visualization.

    fit_method:
      none  = raw connected line
      loess = LOWESS/LOESS-like smoothing
      poly2 = polynomial degree 2
      poly3 = polynomial degree 3
    """
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
            sm = lowess(
                y_ok,
                x_ok,
                frac=float(span),
                return_sorted=True
            )

            x_smooth = sm[:, 0]
            y_smooth = sm[:, 1]

            tmp = pd.DataFrame({"x": x_smooth, "y": y_smooth})
            tmp = tmp.groupby("x", as_index=False)["y"].mean().sort_values("x")

            if tmp.shape[0] >= 2:
                y_grid = np.interp(x_grid, tmp["x"], tmp["y"])
                return x_grid, y_grid

            return x_ok, y_ok

        return x_ok, y_ok

    if fit_method in ["poly2", "poly3"]:
        degree = 2 if fit_method == "poly2" else 3

        if len(np.unique(x_ok)) < degree + 1:
            return x_ok, y_ok

        try:
            coef = np.polyfit(x_ok, y_ok, deg=degree)
            y_grid = np.polyval(coef, x_grid)
            return x_grid, y_grid
        except Exception:
            return x_ok, y_ok

    return x_ok, y_ok


def gene_color_map(genes):
    genes = sorted(list(set(genes)))
    return {
        gene: COLOR_POOL[i % len(COLOR_POOL)]
        for i, gene in enumerate(genes)
    }


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
    show_facet_strip=False,
):
    df = all_tb.copy()

    if "GeneName" not in df.columns:
        df = add_annotation(df)

    df["Time"] = df["Time"].astype(str).str.strip()
    df["Space"] = df["Space"].astype(str).str.strip()

    df["Time"] = pd.Categorical(df["Time"], categories=TIME_LEVELS, ordered=True)
    df["Space"] = pd.Categorical(df["Space"], categories=SPACE_LEVELS, ordered=True)

    df["TimeNum"] = df["Time"].astype(str).map(TIME_NUM_MAP)
    df["SpaceNum"] = df["Space"].astype(str).map(SPACE_NUM_MAP)

    if visualization_mode == "spatial_profile":
        df = df[df["Time"].astype(str) == visualization_time].copy()
        x_cat = "Space"
        x_num = "SpaceNum"
        x_order = SPACE_LEVELS
        x_tickvals = list(range(1, len(SPACE_LEVELS) + 1))
        x_ticktext = SPACE_LEVELS
        x_title = "Location"
        title = f"Clustered spatial fitness profiles at {visualization_time}"
    else:
        df = df[df["Space"].astype(str) == visualization_space].copy()
        x_cat = "Time"
        x_num = "TimeNum"
        x_order = TIME_LEVELS
        x_tickvals = [TIME_NUM_MAP[t] for t in TIME_LEVELS]
        x_ticktext = TIME_LEVELS
        x_title = "Time (h)"
        title = f"Clustered temporal fitness profiles at {visualization_space}"

    df = df.dropna(subset=["Cluster", "Gene", "logFC"]).copy()
    df["Cluster"] = pd.to_numeric(df["Cluster"], errors="coerce")
    df = df.dropna(subset=["Cluster"]).copy()
    df["Cluster"] = df["Cluster"].astype(int)

    df = df.sort_values(["Cluster", "Gene", x_num])

    clusters = sorted(df["Cluster"].dropna().unique().astype(int))
    n_clusters = len(clusters)

    if n_clusters == 0:
        fig = go.Figure()
        fig.update_layout(title="No clusters available")
        return fig

    ncols = 5 if n_clusters >= 10 else min(4, n_clusters)
    nrows = int(np.ceil(n_clusters / ncols))

    subplot_titles = [str(c) for c in clusters]

    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.025,
        vertical_spacing=0.075
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

    for idx, c in enumerate(clusters):
        row = idx // ncols + 1
        col = idx % ncols + 1

        sub = df[df["Cluster"] == c].copy()
        genes = sorted(sub["Gene"].dropna().unique())

        if int(max_genes_per_cluster) <= 0:
            genes_to_plot = []
        elif len(genes) > int(max_genes_per_cluster):
            genes_to_plot = rng.choice(
                genes,
                size=int(max_genes_per_cluster),
                replace=False
            )
        else:
            genes_to_plot = genes

        for gene in genes_to_plot:
            gdf = sub[sub["Gene"] == gene].sort_values(x_num)
            gname = gdf["GeneName"].iloc[0] if "GeneName" in gdf.columns and len(gdf) > 0 else gene

            sx, sy = fit_curve(
                gdf[x_num].values,
                gdf["logFC"].values,
                fit_method=curve_fit_method,
                span=float(loess_span)
            )

            if individual_curve_style == "colored":
                line_color = cmap.get(gene, "rgba(0,150,200,0.9)")
                line_width = 2.0
                opacity = 0.92
            else:
                line_color = "rgba(90,90,90,0.30)"
                line_width = 1.1
                opacity = 1.0

            fig.add_trace(
                go.Scatter(
                    x=sx,
                    y=sy,
                    mode="lines",
                    line=dict(
                        color=line_color,
                        width=line_width,
                        shape="spline" if curve_fit_method != "none" else "linear",
                    ),
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
            med = (
                sub
                .groupby(x_cat, observed=False)["logFC"]
                .median()
                .reindex(x_order)
                .reset_index()
            )

            med[x_num] = med[x_cat].astype(str).map(
                SPACE_NUM_MAP if visualization_mode == "spatial_profile" else TIME_NUM_MAP
            )

            med = med.dropna(subset=[x_num, "logFC"]).sort_values(x_num)

            mx, my = fit_curve(
                med[x_num].values,
                med["logFC"].values,
                fit_method=curve_fit_method,
                span=float(loess_span)
            )

            fig.add_trace(
                go.Scatter(
                    x=mx,
                    y=my,
                    mode="lines",
                    line=dict(
                        color="black",
                        width=3.4,
                        shape="spline" if curve_fit_method != "none" else "linear",
                    ),
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
            gridcolor="rgba(220,220,220,0.85)",
            linecolor="rgba(80,80,80,0.85)",
            mirror=True,
        )

        if fixed_y_axis:
            fig.update_yaxes(
                title_text="logFC" if col == 1 else "",
                range=y_range,
                row=row,
                col=col,
                showgrid=True,
                gridcolor="rgba(220,220,220,0.85)",
                linecolor="rgba(80,80,80,0.85)",
                mirror=True,
            )
        else:
            fig.update_yaxes(
                title_text="logFC" if col == 1 else "",
                row=row,
                col=col,
                showgrid=True,
                gridcolor="rgba(220,220,220,0.85)",
                linecolor="rgba(80,80,80,0.85)",
                mirror=True,
            )

    if show_facet_strip:
        for ann in fig.layout.annotations:
            ann.font = dict(size=14, color="rgba(30,30,30,1)")
            ann.y = ann.y + 0.012

    fig.update_layout(
        title=title,
        template="plotly_white",
        height=max(420, nrows * 285),
        margin=dict(l=55, r=35, t=95, b=70),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(size=13),
    )

    if visualization_mode == "spatial_profile":
        fig.update_xaxes(tickangle=90)
    else:
        fig.update_xaxes(tickangle=0)

    if show_facet_strip:
        for ann in fig.layout.annotations:
            ann.bgcolor = "rgba(215,215,215,1)"
            ann.bordercolor = "rgba(80,80,80,1)"
            ann.borderwidth = 1
            ann.borderpad = 4

    return fig


def make_global_3d_cluster_figure(all_tb):
    df = all_tb.copy()

    if "GeneName" not in df.columns:
        df = add_annotation(df)

    df["Time"] = df["Time"].astype(str)
    df["Space"] = df["Space"].astype(str)

    med = (
        df
        .groupby(["Cluster", "Time", "Space"], observed=False)["logFC"]
        .median()
        .reset_index()
    )

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
        title="3D global clustering: cluster median spatial-temporal fitness profiles",
        template="plotly_white",
        height=760,
        scene=dict(
            xaxis=dict(
                title="Time",
                tickmode="array",
                tickvals=[TIME_NUM_MAP[t] for t in TIME_LEVELS],
                ticktext=TIME_LEVELS,
            ),
            yaxis=dict(
                title="Space",
                tickmode="array",
                tickvals=list(range(1, len(SPACE_LEVELS) + 1)),
                ticktext=SPACE_LEVELS,
            ),
            zaxis=dict(title="Median logFC"),
            camera=dict(eye=dict(x=1.8, y=1.8, z=1.2)),
        ),
        margin=dict(l=0, r=0, t=90, b=0),
    )

    return fig


# ============================================================
# 7. Data result helpers
# ============================================================

def get_cluster_result_from_inputs(
    mode,
    k,
    clustering_method,
    preprocess_options,
    selected_time,
    selected_space,
):
    preprocess_options = preprocess_options or []

    if can_use_precomputed_result(
        mode=mode,
        k=k,
        preprocess_options=preprocess_options,
    ):
        gene_clusters, all_tb, feature_labels = get_precomputed_result(
            clustering_method=clustering_method
        )
        gene_clusters, all_tb = add_active_cluster_columns(
            gene_clusters=gene_clusters,
            all_tb=all_tb,
            clustering_method=clustering_method,
        )
        return gene_clusters.copy(), all_tb.copy(), feature_labels, True

    scale_per_gene = "scale" in preprocess_options

    gene_clusters, all_tb, feature_labels = compute_dual_clustering_cached(
        mode=mode,
        k=int(k),
        scale_per_gene=scale_per_gene,
        selected_time=selected_time,
        selected_space=selected_space,
    )

    gene_clusters, all_tb = add_active_cluster_columns(
        gene_clusters=gene_clusters,
        all_tb=all_tb,
        clustering_method=clustering_method,
    )

    return gene_clusters.copy(), all_tb.copy(), feature_labels, False


def filter_to_selected_clusters(gene_clusters, all_tb, selected_clusters):
    if selected_clusters is None or len(selected_clusters) == 0:
        return gene_clusters.copy(), all_tb.copy()

    selected_clusters = [int(x) for x in selected_clusters]

    gc = gene_clusters[gene_clusters["Cluster"].isin(selected_clusters)].copy()
    tb = all_tb[all_tb["Cluster"].isin(selected_clusters)].copy()

    return gc, tb


def make_cluster_options(all_tb):
    clusters = sorted(
        pd.to_numeric(all_tb["Cluster"], errors="coerce")
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )

    return [{"label": f"Cluster {c}", "value": c} for c in clusters]


def gene_cluster_table_explanation():
    return dbc.Alert(
        [
            html.H5("How to interpret the gene cluster table", className="mb-3"),

            html.P(
                "This table reports gene-level cluster assignments from both DTW distance-based "
                "clustering and cosine distance-based clustering. The currently selected active "
                "method controls the plot, the cluster filter, and the active Cluster/Silhouette columns.",
                className="mb-3",
            ),

            html.Ul(
                [
                    html.Li([
                        html.Strong("Gene: "),
                        "The locus ID or gene ID used as the primary identifier in the fitness dataset."
                    ]),
                    html.Li([
                        html.Strong("GeneName: "),
                        "The annotated gene name from the annotation table. If no gene name is available, "
                        "the gene ID is shown instead."
                    ]),
                    html.Li([
                        html.Strong("Cluster_DTW: "),
                        "The cluster assignment from DTW distance-based clustering. DTW compares trajectory "
                        "shape while allowing local shifts along the spatial-temporal profile."
                    ]),
                    html.Li([
                        html.Strong("Silhouette_DTW: "),
                        "The silhouette score for the DTW-based cluster assignment. Higher values indicate "
                        "that the gene fits better within its assigned DTW cluster."
                    ]),
                    html.Li([
                        html.Strong("Cluster_cosine: "),
                        "The cluster assignment from cosine distance-based clustering. Cosine distance compares "
                        "the overall direction or pattern of the full gene fitness vector."
                    ]),
                    html.Li([
                        html.Strong("Silhouette_cosine: "),
                        "The silhouette score for the cosine-based cluster assignment."
                    ]),
                    html.Li([
                        html.Strong("Cluster: "),
                        "The active cluster assignment currently used for plotting and filtering. If the active "
                        "clustering method is DTW, this equals Cluster_DTW. If the active method is cosine, this "
                        "equals Cluster_cosine."
                    ]),
                    html.Li([
                        html.Strong("Silhouette: "),
                        "The active silhouette score corresponding to the currently selected clustering method."
                    ]),
                    html.Li([
                        html.Strong("Active_Clustering_Method: "),
                        "The clustering method currently selected above and used for the active Cluster and "
                        "Silhouette columns."
                    ]),
                ],
                className="mb-3",
            ),

            html.H6("Silhouette score interpretation", className="mb-2"),
            html.Ul(
                [
                    html.Li([
                        html.Strong("Close to 1: "),
                        "The gene is well matched to its assigned cluster and is far from other clusters."
                    ]),
                    html.Li([
                        html.Strong("Around 0: "),
                        "The gene lies near the boundary between clusters. Its assignment is less distinct."
                    ]),
                    html.Li([
                        html.Strong("Negative: "),
                        "The gene may fit better in another cluster than in its assigned cluster."
                    ]),
                    html.Li([
                        html.Strong("Blank / NA: "),
                        "The precomputed clustering table does not contain silhouette scores for that method. "
                        "When clustering is computed on demand, the app calculates silhouette scores directly."
                    ]),
                ],
                className="mb-0",
            ),
        ],
        color="secondary",
        className="mb-3",
    )


# ============================================================
# 8. Layout
# ============================================================

layout = dbc.Container(
    [
        html.H2("Clustering", className="page-title"),

        html.P(
            "Cluster genes based on their spatial, temporal, or global spatial-temporal "
            "in vivo fitness profiles. This page supports cosine distance-based clustering "
            "and DTW distance-based clustering.",
            className="lead"
        ),

        dbc.Alert(
            [
                html.Strong("How to use this page: "),
                html.Span(
                    "This page clusters genes based on their in vivo fitness profiles across intestinal space, "
                    "infection time, or the full spatial-temporal matrix. By default, it displays the precomputed "
                    "3D global DTW clustering result with 20 clusters. You can switch between DTW and cosine "
                    "distance-based clustering, choose 2D spatial clustering, 2D temporal clustering, or 3D global "
                    "clustering, and visualize the resulting clusters as spatial profiles, temporal profiles, or "
                    "3D median profiles. Use the cluster filter to display only specific clusters."
                ),
                html.Br(),
                html.Br(),
                html.Strong("DTW distance-based clustering: "),
                html.Span(
                    "Dynamic Time Warping compares the shape of fitness trajectories while allowing local shifts "
                    "along the profile. It is useful when two genes show similar patterns but the peak, valley, or "
                    "transition occurs at slightly different spatial positions or time points."
                ),
                html.Br(),
                html.Strong("Cosine distance-based clustering: "),
                html.Span(
                    "Cosine distance compares the overall direction of fitness-profile vectors. It is useful for "
                    "grouping genes with similar global profile patterns, but it is less flexible than DTW when "
                    "patterns are locally shifted."
                ),
            ],
            color="secondary",
            className="mb-4",
        ),

        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Label("Clustering mode"),
                        dcc.Dropdown(
                            id="cluster-mode",
                            options=[
                                {"label": "2D spatial clustering", "value": "spatial"},
                                {"label": "2D temporal clustering", "value": "temporal"},
                                {"label": "3D global clustering", "value": "global"},
                            ],
                            value=DEFAULT_CLUSTERING_MODE,
                            clearable=False,
                        ),
                    ],
                    md=3,
                ),
                dbc.Col(
                    [
                        html.Label("Number of clusters"),
                        dbc.Input(
                            id="cluster-k",
                            type="number",
                            min=2,
                            max=100,
                            step=1,
                            value=DEFAULT_K,
                        ),
                    ],
                    md=2,
                ),
                dbc.Col(
                    [
                        html.Label("Active clustering method"),
                        dcc.Dropdown(
                            id="cluster-method",
                            options=[
                                {"label": "DTW distance-based clustering", "value": "dtw"},
                                {"label": "Cosine distance-based clustering", "value": "cosine"},
                            ],
                            value=DEFAULT_CLUSTERING_METHOD,
                            clearable=False,
                        ),
                    ],
                    md=3,
                ),
                dbc.Col(
                    [
                        html.Label("Preprocessing options"),
                        dcc.Checklist(
                            id="cluster-preprocess-options",
                            options=[
                                {"label": "Z-score scale each gene before clustering", "value": "scale"},
                            ],
                            value=[],
                            inline=False,
                        ),
                    ],
                    md=4,
                ),
            ],
            className="mb-3",
        ),

        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Label("Time point used for 2D spatial clustering"),
                        dcc.Dropdown(
                            id="cluster-selected-time",
                            options=[{"label": t, "value": t} for t in TIME_LEVELS],
                            value="12h",
                            clearable=False,
                        ),
                    ],
                    md=3,
                ),
                dbc.Col(
                    [
                        html.Label("Space point used for 2D temporal clustering"),
                        dcc.Dropdown(
                            id="cluster-selected-space",
                            options=[{"label": s, "value": s} for s in SPACE_LEVELS],
                            value="SI8",
                            clearable=False,
                        ),
                    ],
                    md=3,
                ),
                dbc.Col(
                    [
                        html.Label("Show selected clusters only"),
                        dcc.Dropdown(
                            id="cluster-filter",
                            options=[],
                            value=[],
                            multi=True,
                            placeholder="Leave empty to show all clusters",
                        ),
                    ],
                    md=6,
                ),
            ],
            className="mb-4",
        ),

        html.Hr(),

        html.H4("Cluster visualization"),

        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Label("Visualization type"),
                        dcc.Dropdown(
                            id="cluster-visualization-mode",
                            options=[
                                {
                                    "label": "Spatial profiles at selected time point",
                                    "value": "spatial_profile",
                                },
                                {
                                    "label": "Temporal profiles at selected space point",
                                    "value": "temporal_profile",
                                },
                                {
                                    "label": "3D cluster median profiles",
                                    "value": "global_3d",
                                },
                            ],
                            value="spatial_profile",
                            clearable=False,
                        ),
                    ],
                    md=3,
                ),
                dbc.Col(
                    [
                        html.Label("Time point for spatial-profile visualization"),
                        dcc.Dropdown(
                            id="cluster-visualization-time",
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
                            id="cluster-visualization-space",
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
                            id="cluster-max-curves",
                            type="number",
                            min=0,
                            max=1000,
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
                            id="cluster-plot-options",
                            options=[
                                {"label": "Show cluster median curve", "value": "median"},
                                {"label": "Use fixed Y-axis across clusters", "value": "fixed_y"},
                                {"label": "Show R-like facet strip labels", "value": "facet_strip"},
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
                            id="cluster-individual-curve-style",
                            options=[
                                {"label": "Colored individual curves", "value": "colored"},
                                {"label": "Grey individual curves", "value": "grey"},
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
                            id="cluster-curve-fit-method",
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
                        html.Div(
                            id="cluster-loess-span-wrapper",
                            children=[
                                html.Label("LOESS span"),
                                dbc.Input(
                                    id="cluster-loess-span",
                                    type="number",
                                    min=0.1,
                                    max=1.0,
                                    step=0.05,
                                    value=0.8,
                                ),
                                html.Small(
                                    "Equivalent to ggplot2 span. Default = 0.8.",
                                    className="text-muted"
                                ),
                            ],
                        ),
                    ],
                    md=3,
                ),
            ],
            className="mb-4",
        ),

        dcc.Tabs(
            id="cluster-tabs",
            value="plot",
            children=[
                dcc.Tab(label="Cluster Plot", value="plot"),
                dcc.Tab(label="Gene Cluster Table", value="gene_table"),
            ],
        ),

        html.Br(),

        dcc.Loading(
            type="circle",
            children=[
                html.Div(id="cluster-summary"),
                html.Div(id="cluster-content"),
            ],
        ),

        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Button(
                            "Download gene cluster table",
                            id="download-gene-cluster-button",
                            color="secondary",
                            size="sm",
                            className="mt-3 me-2",
                        ),
                        dcc.Download(id="download-gene-cluster"),
                    ],
                    md=4,
                ),
            ],
            className="mb-5",
        ),
    ],
    fluid=True,
)


# ============================================================
# 9. Callbacks
# ============================================================

@dash.callback(
    Output("cluster-loess-span-wrapper", "style"),
    Input("cluster-curve-fit-method", "value"),
)
def toggle_loess_span(curve_fit_method):
    if curve_fit_method == "loess":
        return {"display": "block"}

    return {"display": "none"}


@dash.callback(
    Output("cluster-filter", "options"),
    Output("cluster-filter", "value"),
    Input("cluster-mode", "value"),
    Input("cluster-k", "value"),
    Input("cluster-method", "value"),
    Input("cluster-preprocess-options", "value"),
    Input("cluster-selected-time", "value"),
    Input("cluster-selected-space", "value"),
    State("cluster-filter", "value"),
)
def update_cluster_filter_options(
    mode,
    k,
    clustering_method,
    preprocess_options,
    selected_time,
    selected_space,
    current_value,
):
    try:
        gene_clusters, all_tb, feature_labels, used_precomputed = get_cluster_result_from_inputs(
            mode=mode,
            k=k,
            clustering_method=clustering_method,
            preprocess_options=preprocess_options,
            selected_time=selected_time,
            selected_space=selected_space,
        )

        options = make_cluster_options(all_tb)
        allowed = {opt["value"] for opt in options}

        current_value = current_value or []
        kept_value = [int(v) for v in current_value if int(v) in allowed]

        return options, kept_value

    except Exception:
        return [], []


@dash.callback(
    Output("cluster-summary", "children"),
    Output("cluster-content", "children"),
    Input("cluster-tabs", "value"),
    Input("cluster-mode", "value"),
    Input("cluster-k", "value"),
    Input("cluster-method", "value"),
    Input("cluster-preprocess-options", "value"),
    Input("cluster-selected-time", "value"),
    Input("cluster-selected-space", "value"),
    Input("cluster-filter", "value"),
    Input("cluster-visualization-mode", "value"),
    Input("cluster-visualization-time", "value"),
    Input("cluster-visualization-space", "value"),
    Input("cluster-max-curves", "value"),
    Input("cluster-plot-options", "value"),
    Input("cluster-individual-curve-style", "value"),
    Input("cluster-curve-fit-method", "value"),
    Input("cluster-loess-span", "value"),
)
def update_clustering_page(
    active_tab,
    mode,
    k,
    clustering_method,
    preprocess_options,
    selected_time,
    selected_space,
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
        plot_options = plot_options or []
        selected_clusters = selected_clusters or []
        loess_span = float(loess_span or 0.8)

        gene_clusters, all_tb, feature_labels, used_precomputed = get_cluster_result_from_inputs(
            mode=mode,
            k=k,
            clustering_method=clustering_method,
            preprocess_options=preprocess_options,
            selected_time=selected_time,
            selected_space=selected_space,
        )

        gene_clusters_filtered, all_tb_filtered = filter_to_selected_clusters(
            gene_clusters=gene_clusters,
            all_tb=all_tb,
            selected_clusters=selected_clusters,
        )

        show_median = "median" in plot_options
        fixed_y_axis = "fixed_y" in plot_options
        show_facet_strip = "facet_strip" in plot_options

        summary = summarize_clustering(
            gene_clusters=gene_clusters_filtered,
            mode=mode,
            k=k,
            clustering_method=clustering_method,
            selected_time=selected_time,
            selected_space=selected_space,
            visualization_mode=visualization_mode,
            visualization_time=visualization_time,
            visualization_space=visualization_space,
            fixed_y_axis=fixed_y_axis,
            curve_fit_method=curve_fit_method,
            loess_span=loess_span,
            individual_curve_style=individual_curve_style,
            show_median=show_median,
            used_precomputed=used_precomputed,
            selected_clusters=selected_clusters,
        )

        if active_tab == "plot":
            if visualization_mode in ["spatial_profile", "temporal_profile"]:
                fig = make_faceted_profile_figure(
                    all_tb=all_tb_filtered,
                    visualization_mode=visualization_mode,
                    visualization_time=visualization_time,
                    visualization_space=visualization_space,
                    max_genes_per_cluster=int(max_curves),
                    show_median=show_median,
                    curve_fit_method=curve_fit_method,
                    loess_span=loess_span,
                    fixed_y_axis=fixed_y_axis,
                    individual_curve_style=individual_curve_style,
                    show_facet_strip=show_facet_strip,
                )
            else:
                fig = make_global_3d_cluster_figure(all_tb_filtered)

            content = dcc.Graph(figure=fig, style={"width": "100%"})
            return summary, content

        # Gene cluster table only
        df = gene_clusters_filtered.copy()

        table_cols = [
            "Gene",
            "GeneName",
            "Cluster_DTW",
            "Silhouette_DTW",
            "Cluster_cosine",
            "Silhouette_cosine",
            "Cluster",
            "Silhouette",
            "Active_Clustering_Method",
        ]

        existing_cols = [c for c in table_cols if c in df.columns]
        df = df[existing_cols].copy()

        for col in ["Silhouette_DTW", "Silhouette_cosine", "Silhouette"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").round(4)

        content = html.Div(
            [
                gene_cluster_table_explanation(),

                dash_table.DataTable(
                    data=df.to_dict("records"),
                    columns=[{"name": c, "id": c} for c in df.columns],
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
                    style_header={
                        "fontWeight": "bold",
                        "backgroundColor": "#f1f3f5",
                    },
                ),
            ]
        )

        return summary, content

    except Exception as e:
        return "", dbc.Alert(str(e), color="danger")


@dash.callback(
    Output("download-gene-cluster", "data"),
    Input("download-gene-cluster-button", "n_clicks"),
    State("cluster-mode", "value"),
    State("cluster-k", "value"),
    State("cluster-method", "value"),
    State("cluster-preprocess-options", "value"),
    State("cluster-selected-time", "value"),
    State("cluster-selected-space", "value"),
    State("cluster-filter", "value"),
    prevent_initial_call=True,
)
def download_gene_cluster_table(
    n_clicks,
    mode,
    k,
    clustering_method,
    preprocess_options,
    selected_time,
    selected_space,
    selected_clusters,
):
    if not n_clicks:
        return no_update

    gene_clusters, all_tb, feature_labels, used_precomputed = get_cluster_result_from_inputs(
        mode=mode,
        k=k,
        clustering_method=clustering_method,
        preprocess_options=preprocess_options,
        selected_time=selected_time,
        selected_space=selected_space,
    )

    gene_clusters_filtered, all_tb_filtered = filter_to_selected_clusters(
        gene_clusters=gene_clusters,
        all_tb=all_tb,
        selected_clusters=selected_clusters or [],
    )

    df = gene_clusters_filtered.copy()

    table_cols = [
        "Gene",
        "GeneName",
        "Cluster_DTW",
        "Silhouette_DTW",
        "Cluster_cosine",
        "Silhouette_cosine",
        "Cluster",
        "Silhouette",
        "Active_Clustering_Method",
    ]

    existing_cols = [c for c in table_cols if c in df.columns]
    df = df[existing_cols].copy()

    source_label = "precomputed" if used_precomputed else "computed"

    if selected_clusters:
        cluster_label = "clusters_" + "_".join([str(x) for x in selected_clusters])
    else:
        cluster_label = "all_clusters"

    filename = (
        f"gene_cluster_table_{source_label}_{mode}_k{k}_"
        f"active_{clustering_method}_{cluster_label}.csv"
    )

    return dcc.send_data_frame(df.to_csv, filename, index=False)
