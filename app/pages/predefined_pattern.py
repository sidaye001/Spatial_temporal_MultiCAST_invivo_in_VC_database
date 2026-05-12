import os
import io
import base64
from functools import lru_cache

import numpy as np
import pandas as pd
import dash
from dash import html, dcc, dash_table, Input, Output, State, no_update
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    from statsmodels.nonparametric.smoothers_lowess import lowess
    HAS_LOWESS = True
except Exception:
    HAS_LOWESS = False


dash.register_page(__name__, path="/predefined-pattern", name="Predefined Pattern", order=6)


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

TIME_LEVELS = ["1h", "3h", "6h", "12h", "24h"]
SPACE_LEVELS = [
    "st", "SI1", "SI2", "SI3", "SI4", "SI5",
    "SI6", "SI7", "SI8", "SI9", "ce", "co"
]

DEFAULT_START_TIME = "12h"
DEFAULT_END_TIME = "24h"
DEFAULT_START_SPACE = "SI6"
DEFAULT_END_SPACE = "co"
DEFAULT_TOP_N = 30
DEFAULT_DTW_WINDOW_FRACTION = 0.05

# R-script default pattern:
# Feature order is time_then_space:
# 12h_SI6, 12h_SI7, 12h_SI8, 12h_SI9, 12h_ce, 12h_co,
# 24h_SI6, 24h_SI7, 24h_SI8, 24h_SI9, 24h_ce, 24h_co
DEFAULT_CUSTOM_TREND_VALUES = [
    0.00, 0.00, 0.00, -0.35, -2.00, -4.00,
    0.00, 0.00, 0.00, -0.55, -2.50, -5.00,
]

DEFAULT_MATRIX_TEXT = """0, 0, 0, -0.35, -2, -4
0, 0, 0, -0.55, -2.5, -5"""

BROWN_BLUE_SCALE = [
    [0.00, "#08306B"],
    [0.20, "#2171B5"],
    [0.40, "#6BAED6"],
    [0.49, "#FFFFFF"],
    [0.51, "#FFFFFF"],
    [0.60, "#FDBB84"],
    [0.80, "#E34A33"],
    [1.00, "#7F0000"],
]

TIME_COLOR_MAP = {
    "1h": "#1f77b4",
    "3h": "#2ca02c",
    "6h": "#9467bd",
    "12h": "#ff7f0e",
    "24h": "#d62728",
}


# ============================================================
# 1. Data loading
# ============================================================

@lru_cache(maxsize=1)
def load_raw_data():
    if not os.path.exists(RAW_FILE):
        raise FileNotFoundError(f"Raw data file not found: {RAW_FILE}")

    df = pd.read_csv(RAW_FILE)
    required = {"Gene", "Time", "Space"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Raw data file is missing required columns: {missing}")

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
    df["Time"] = pd.Categorical(df["Time"], categories=TIME_LEVELS, ordered=True)
    df["Space"] = pd.Categorical(df["Space"], categories=SPACE_LEVELS, ordered=True)

    return df


@lru_cache(maxsize=1)
def load_annotation():
    raw = load_raw_data()
    genes = sorted(raw["Gene"].dropna().astype(str).unique())

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
    ann = load_annotation()
    out = df.copy()
    for col in ["GeneName", "gene_name", "GeneName_x", "GeneName_y"]:
        if col in out.columns:
            out = out.drop(columns=[col])
    out["Gene"] = out["Gene"].astype(str)
    out = out.merge(ann, on="Gene", how="left")
    out["GeneName"] = out["GeneName"].fillna(out["Gene"])
    return out


# ============================================================
# 2. Helper functions
# ============================================================

def get_ordered_window(levels, start_value, end_value):
    if start_value not in levels:
        raise ValueError(f"Start value not found: {start_value}")
    if end_value not in levels:
        raise ValueError(f"End value not found: {end_value}")

    i1 = levels.index(start_value)
    i2 = levels.index(end_value)
    if i1 <= i2:
        return levels[i1:(i2 + 1)]
    return levels[i2:(i1 + 1)]


def zscore_vec(v):
    v = np.asarray(v, dtype=float)
    m = np.nanmean(v)
    s = np.nanstd(v, ddof=1)
    if not np.isfinite(s) or s == 0:
        return np.zeros_like(v, dtype=float)
    return (v - m) / s


def make_feature_order(timepoints, spacepoints):
    # Explicit time_then_space order, matching the uploaded R script.
    return [f"{t}_{s}" for t in timepoints for s in spacepoints]


def make_default_trend_values(timepoints, spacepoints):
    default_times = get_ordered_window(TIME_LEVELS, DEFAULT_START_TIME, DEFAULT_END_TIME)
    default_spaces = get_ordered_window(SPACE_LEVELS, DEFAULT_START_SPACE, DEFAULT_END_SPACE)

    if list(timepoints) == default_times and list(spacepoints) == default_spaces:
        return DEFAULT_CUSTOM_TREND_VALUES.copy()

    # Generic fallback for non-default windows: decreasing along the feature axis.
    n = len(timepoints) * len(spacepoints)
    return np.linspace(1, -1, n).tolist()


def make_pattern_table(timepoints, spacepoints, values=None):
    feature_order = make_feature_order(timepoints, spacepoints)

    if values is None:
        values = make_default_trend_values(timepoints, spacepoints)

    if len(values) != len(feature_order):
        values = make_default_trend_values(timepoints, spacepoints)

    rows = []
    k = 0
    for t in timepoints:
        row = {"Time": t}
        for s in spacepoints:
            row[s] = float(values[k])
            k += 1
        rows.append(row)
    return rows


def table_data_to_values(table_data, timepoints, spacepoints):
    values = []
    for row in table_data or []:
        for s in spacepoints:
            val = row.get(s, 0)
            try:
                val = float(val)
            except Exception:
                val = 0.0
            if not np.isfinite(val):
                val = 0.0
            values.append(val)
    expected = len(timepoints) * len(spacepoints)
    if len(values) != expected:
        raise ValueError(f"Pattern table has {len(values)} values; expected {expected}.")
    return values


def parse_matrix_text(matrix_text, timepoints, spacepoints):
    if matrix_text is None or str(matrix_text).strip() == "":
        raise ValueError("Matrix input is empty. Enter one row per timepoint and one column per spacepoint.")

    text = str(matrix_text).strip()
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if line == "":
            continue
        for sep in ["\t", ";"]:
            line = line.replace(sep, ",")
        parts = [x.strip() for x in line.split(",") if x.strip() != ""]
        if len(parts) == 1:
            parts = [x.strip() for x in line.split(" ") if x.strip() != ""]
        rows.append([float(x) for x in parts])

    n_time = len(timepoints)
    n_space = len(spacepoints)

    if len(rows) == 1 and len(rows[0]) == n_time * n_space:
        return [float(x) for x in rows[0]]

    if len(rows) != n_time:
        raise ValueError(f"Matrix input must have {n_time} rows, one for each selected timepoint.")

    values = []
    for i, row in enumerate(rows):
        if len(row) != n_space:
            raise ValueError(
                f"Matrix row {i + 1} has {len(row)} values; expected {n_space} values for spaces: "
                f"{', '.join(spacepoints)}."
            )
        values.extend(row)
    return values


def build_gene_feature_matrix(timepoints, spacepoints):
    raw = load_raw_data()
    genes = sorted(raw["Gene"].dropna().astype(str).unique())
    feature_order = make_feature_order(timepoints, spacepoints)

    sub = raw[
        raw["Time"].astype(str).isin(timepoints) &
        raw["Space"].astype(str).isin(spacepoints)
    ].copy()

    all_index = pd.MultiIndex.from_product(
        [genes, timepoints, spacepoints],
        names=["Gene", "Time", "Space"]
    )

    feature_df = (
        sub.groupby(["Gene", "Time", "Space"], observed=False)["logFC"]
        .mean()
        .reindex(all_index)
        .reset_index()
    )
    feature_df["logFC"] = pd.to_numeric(feature_df["logFC"], errors="coerce").fillna(0.0)
    feature_df["Feature"] = feature_df["Time"].astype(str) + "_" + feature_df["Space"].astype(str)

    xdf = (
        feature_df.pivot_table(index="Gene", columns="Feature", values="logFC", aggfunc="mean")
        .reindex(index=genes, columns=feature_order)
        .fillna(0.0)
        .reset_index()
    )

    X_raw = xdf[feature_order].to_numpy(dtype=float)
    X_raw[~np.isfinite(X_raw)] = 0.0

    return xdf, X_raw, feature_order


def dtw_distance(vec_a, vec_b, window_fraction=0.05):
    a = np.asarray(vec_a, dtype=float)
    b = np.asarray(vec_b, dtype=float)
    n = len(a)
    m = len(b)
    window = max(abs(n - m), max(1, int(round(float(window_fraction) * max(n, m)))))

    dp = np.full((n + 1, m + 1), np.inf)
    dp[0, 0] = 0.0

    for i in range(1, n + 1):
        j_start = max(1, i - window)
        j_end = min(m, i + window)
        for j in range(j_start, j_end + 1):
            cost = abs(a[i - 1] - b[j - 1])
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])

    return float(dp[n, m])


def compute_trend_search(
    timepoints,
    spacepoints,
    trend_raw_values,
    scale_gene_vector=True,
    scale_trend_vector=True,
    dtw_window_fraction=0.05,
):
    xdf, X_raw, feature_order = build_gene_feature_matrix(timepoints, spacepoints)

    trend_raw = np.asarray(trend_raw_values, dtype=float)
    if len(trend_raw) != len(feature_order):
        raise ValueError(f"Trend vector length must be {len(feature_order)}, got {len(trend_raw)}.")

    X_for_distance = X_raw.copy()
    if scale_gene_vector:
        X_for_distance = np.apply_along_axis(zscore_vec, 1, X_for_distance)

    trend_for_dtw = trend_raw.copy()
    if scale_trend_vector:
        trend_for_dtw = zscore_vec(trend_for_dtw)

    distances = []
    for i in range(X_for_distance.shape[0]):
        distances.append(dtw_distance(X_for_distance[i, :], trend_for_dtw, dtw_window_fraction))

    distances = np.asarray(distances, dtype=float)

    spearman_values = []
    trend_series = pd.Series(trend_for_dtw)
    for i in range(X_for_distance.shape[0]):
        try:
            corr = pd.Series(X_for_distance[i, :]).corr(trend_series, method="spearman")
        except Exception:
            corr = np.nan
        spearman_values.append(corr)

    rank_table = pd.DataFrame({
        "Gene": xdf["Gene"].astype(str).values,
        "DTW_distance_to_trend": distances,
        "Spearman_correlation_to_trend": spearman_values,
    })
    rank_table = rank_table.sort_values("DTW_distance_to_trend", ascending=True).reset_index(drop=True)
    rank_table["Rank"] = np.arange(1, len(rank_table) + 1)
    rank_table = add_annotation(rank_table)

    rank_table["Selected_Timepoints"] = ",".join(timepoints)
    rank_table["Selected_Spacepoints"] = ",".join(spacepoints)
    rank_table["Scale_Gene_Vector"] = bool(scale_gene_vector)
    rank_table["Scale_Trend_Vector"] = bool(scale_trend_vector)
    rank_table["DTW_Window_Fraction"] = float(dtw_window_fraction)

    rank_table = rank_table[[
        "Rank", "Gene", "GeneName", "DTW_distance_to_trend", "Spearman_correlation_to_trend",
        "Selected_Timepoints", "Selected_Spacepoints",
        "Scale_Gene_Vector", "Scale_Trend_Vector", "DTW_Window_Fraction"
    ]]

    trend_table = pd.DataFrame({
        "Feature": feature_order,
        "Trend_Raw": trend_raw,
        "Trend_For_DTW": trend_for_dtw,
    })
    trend_table[["Time", "Space"]] = trend_table["Feature"].str.split("_", n=1, expand=True)
    trend_table["Feature_Index"] = np.arange(1, len(trend_table) + 1)

    feature_matrix_out = xdf.copy()
    feature_matrix_out = feature_matrix_out.merge(
        rank_table[["Gene", "GeneName", "Rank", "DTW_distance_to_trend", "Spearman_correlation_to_trend"]],
        on="Gene",
        how="left",
    )

    leading_cols = ["Gene", "GeneName", "Rank", "DTW_distance_to_trend", "Spearman_correlation_to_trend"]
    feature_matrix_out = feature_matrix_out[leading_cols + feature_order]

    return rank_table, trend_table, feature_matrix_out, X_for_distance, feature_order


# ============================================================
# 3. Plot functions
# ============================================================

def make_trend_3d_figure(trend_table, timepoints, spacepoints, use_scaled=True):
    value_col = "Trend_For_DTW" if use_scaled else "Trend_Raw"
    zdf = trend_table.pivot(index="Time", columns="Space", values=value_col).reindex(index=timepoints, columns=spacepoints)
    z = zdf.to_numpy(dtype=float)

    fig = go.Figure()
    fig.add_trace(
        go.Surface(
            x=list(range(1, len(spacepoints) + 1)),
            y=list(range(1, len(timepoints) + 1)),
            z=z,
            colorscale=BROWN_BLUE_SCALE,
            opacity=0.88,
            colorbar=dict(
                title=value_col,
                x=0.82,
                xanchor="left",
                len=0.68,
                thickness=16,
                y=0.52,
            ),
            hovertemplate="Space index: %{x}<br>Time index: %{y}<br>Value: %{z:.3f}<extra></extra>",
            name="Predesigned trend",
            showscale=True,
        )
    )

    for i, tt in enumerate(timepoints, start=1):
        line_df = trend_table[trend_table["Time"] == tt].copy()
        line_df["Space_Index"] = line_df["Space"].map({s: k + 1 for k, s in enumerate(spacepoints)})
        line_df = line_df.sort_values("Space_Index")
        time_color = TIME_COLOR_MAP.get(str(tt), "#ff7f0e")
        fig.add_trace(
            go.Scatter3d(
                x=line_df["Space_Index"],
                y=[i] * len(line_df),
                z=line_df[value_col],
                mode="markers+lines",
                marker=dict(size=6, color=time_color),
                line=dict(width=7, color=time_color),
                text=[
                    f"Feature: {r.Feature}<br>Trend raw: {r.Trend_Raw:.3f}<br>Trend for DTW: {r.Trend_For_DTW:.3f}"
                    for r in line_df.itertuples()
                ],
                hoverinfo="text",
                name=f"Trend {tt}",
                showlegend=True,
            )
        )

    fig.update_layout(
        title="3D predefined spatial-temporal trend pattern",
        template="plotly_white",
        height=680,
        legend=dict(
            x=0.98,
            y=0.96,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.75)",
            bordercolor="rgba(0,0,0,0.12)",
            borderwidth=1,
        ),
        scene=dict(
            domain=dict(x=[0.0, 0.76], y=[0.0, 1.0]),
            xaxis=dict(
                title="Space",
                tickmode="array",
                tickvals=list(range(1, len(spacepoints) + 1)),
                ticktext=spacepoints,
            ),
            yaxis=dict(
                title="Time",
                tickmode="array",
                tickvals=list(range(1, len(timepoints) + 1)),
                ticktext=timepoints,
            ),
            zaxis=dict(title="Scaled trend" if use_scaled else "Raw trend"),
            camera=dict(eye=dict(x=1.6, y=1.5, z=1.1)),
        ),
        margin=dict(l=10, r=170, t=80, b=20),
    )
    return fig


def smooth_curve(x, y, span=0.8):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 3 or not HAS_LOWESS:
        return x, y
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    smoothed = lowess(y, x, frac=float(span), return_sorted=True)
    return smoothed[:, 0], smoothed[:, 1]


def make_top_gene_curve_figure(
    rank_table,
    trend_table,
    X_for_distance,
    feature_order,
    top_n=30,
    curve_style="loess",
    loess_span=0.8,
):
    top_n = max(1, int(top_n))
    top_genes = rank_table.sort_values("Rank").head(top_n)["Gene"].astype(str).tolist()

    gene_order = rank_table["Gene"].astype(str).tolist()
    gene_to_i = {g: i for i, g in enumerate(gene_order)}
    top_genes = [g for g in top_genes if g in gene_to_i]

    x = np.arange(1, len(feature_order) + 1, dtype=float)
    trend_y = trend_table.sort_values("Feature_Index")["Trend_For_DTW"].to_numpy(dtype=float)

    fig = go.Figure()

    for gid in top_genes:
        i = gene_to_i[gid]
        y = X_for_distance[i, :]
        row = rank_table.loc[rank_table["Gene"] == gid].iloc[0]
        label = f"{int(row['Rank'])}. {row['GeneName']} | {gid} | DTW={row['DTW_distance_to_trend']:.3f}"

        if curve_style == "loess":
            xs, ys = smooth_curve(x, y, span=loess_span)
            mode = "lines"
        else:
            xs, ys = x, y
            mode = "lines+markers"

        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode=mode,
                line=dict(color="rgba(130,130,130,0.52)", width=1.0),
                marker=dict(size=4, color="rgba(130,130,130,0.50)"),
                hovertemplate=(
                    f"{label}<br>Feature index: %{{x}}<br>Value: %{{y:.3f}}<extra></extra>"
                ),
                name=label,
                showlegend=False,
            )
        )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=trend_y,
            mode="lines+markers",
            line=dict(color="black", width=3.2),
            marker=dict(size=7, color="black"),
            text=feature_order,
            hovertemplate="Trend<br>Feature: %{text}<br>Value: %{y:.3f}<extra></extra>",
            name="Predefined trend",
            showlegend=True,
        )
    )

    fig.update_layout(
        title=f"Top {len(top_genes)} genes ranked by DTW distance to predefined trend",
        template="plotly_white",
        height=650,
        xaxis=dict(
            title="Selected time-space feature",
            tickmode="array",
            tickvals=list(range(1, len(feature_order) + 1)),
            ticktext=feature_order,
            tickangle=90,
        ),
        yaxis=dict(title="Scaled logFC / scaled trend"),
        margin=dict(l=60, r=30, t=80, b=140),
    )
    return fig


def make_top_heatmap_figure(rank_table, feature_matrix, feature_order, top_n=30):
    """
    Draw the top-gene heatmap using the original raw Beta/logFC values.

    Important: DTW ranking can use scaled vectors, but the heatmap should show
    the biological/raw values from feature_matrix. This prevents values such as
    dcuA 24h_co from appearing as z-scored values instead of raw Beta/logFC.
    """
    top = rank_table.sort_values("Rank").head(int(top_n)).copy()

    fm = feature_matrix.copy()
    fm["Gene"] = fm["Gene"].astype(str)
    fm = fm.set_index("Gene", drop=False)

    rows = []
    labels = []
    customdata = []

    for _, row in top.iterrows():
        gid = str(row["Gene"])
        if gid not in fm.index:
            continue

        vals = pd.to_numeric(fm.loc[gid, feature_order], errors="coerce").to_numpy(dtype=float)
        rows.append(vals)

        label = f"{int(row['Rank'])}. {row['GeneName']} | {gid} | DTW={row['DTW_distance_to_trend']:.3f}"
        labels.append(label)
        customdata.append([str(row["GeneName"]), gid] * len(feature_order))

    if not rows:
        return go.Figure()

    z = np.vstack(rows)

    # customdata shape: n_genes x n_features x 2
    cd = np.empty((len(labels), len(feature_order), 2), dtype=object)
    for i, (_, row) in enumerate(top.head(len(labels)).iterrows()):
        cd[i, :, 0] = str(row["GeneName"])
        cd[i, :, 1] = str(row["Gene"])

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=feature_order,
            y=labels,
            customdata=cd,
            colorscale=BROWN_BLUE_SCALE,
            zmid=0,
            colorbar=dict(title="Raw Beta/logFC"),
            hovertemplate=(
                "GeneName: %{customdata[0]}<br>"
                "Gene ID: %{customdata[1]}<br>"
                "Feature: %{x}<br>"
                "Raw Beta/logFC: %{z:.3f}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title=f"Top {len(labels)} genes heatmap ranked by DTW distance; color = raw Beta/logFC",
        template="plotly_white",
        height=max(520, 24 * len(labels) + 180),
        xaxis=dict(title="Selected time-space feature", tickangle=90),
        yaxis=dict(title="Gene", autorange="reversed"),
        margin=dict(l=260, r=70, t=80, b=140),
    )
    return fig


def make_download_csv(df):
    return df.to_csv(index=False)


# ============================================================
# 4. Layout
# ============================================================

layout = dbc.Container(
    [
        html.H2("Predefined Spatial-Temporal Pattern Search", className="page-title"),
        html.P(
            "Define a spatial-temporal trend pattern across selected infection timepoints and GI locations, "
            "visualize the 3D predefined pattern, and rank all genes by DTW distance to this pattern.",
            className="lead",
        ),

        dbc.Alert(
            [
                html.Strong("Default pattern: "),
                html.Span(
                    "12h and 24h across SI6-co; SI6-SI8 are flat, SI9 starts dropping, "
                    "and ce/co drop strongly. Genes are ranked only by DTW distance to the predefined trend."
                ),
            ],
            color="secondary",
            className="mb-3",
        ),

        dbc.Row(
            [
                dbc.Col([
                    html.Label("Start time"),
                    dcc.Dropdown(
                        id="trend-start-time",
                        options=[{"label": x, "value": x} for x in TIME_LEVELS],
                        value=DEFAULT_START_TIME,
                        clearable=False,
                    ),
                ], md=2),
                dbc.Col([
                    html.Label("End time"),
                    dcc.Dropdown(
                        id="trend-end-time",
                        options=[{"label": x, "value": x} for x in TIME_LEVELS],
                        value=DEFAULT_END_TIME,
                        clearable=False,
                    ),
                ], md=2),
                dbc.Col([
                    html.Label("Start space"),
                    dcc.Dropdown(
                        id="trend-start-space",
                        options=[{"label": x, "value": x} for x in SPACE_LEVELS],
                        value=DEFAULT_START_SPACE,
                        clearable=False,
                    ),
                ], md=2),
                dbc.Col([
                    html.Label("End space"),
                    dcc.Dropdown(
                        id="trend-end-space",
                        options=[{"label": x, "value": x} for x in SPACE_LEVELS],
                        value=DEFAULT_END_SPACE,
                        clearable=False,
                    ),
                ], md=2),
                dbc.Col([
                    html.Label("Top N genes"),
                    dbc.Input(
                        id="trend-top-n",
                        type="number",
                        min=1,
                        max=200,
                        step=1,
                        value=DEFAULT_TOP_N,
                    ),
                ], md=2),
                dbc.Col([
                    html.Label("DTW window fraction"),
                    dbc.Input(
                        id="trend-dtw-window-fraction",
                        type="number",
                        min=0,
                        max=1,
                        step=0.01,
                        value=DEFAULT_DTW_WINDOW_FRACTION,
                    ),
                ], md=2),
            ],
            className="mb-3",
        ),

        dbc.Row(
            [
                dbc.Col([
                    html.Label("Pattern input mode"),
                    dcc.RadioItems(
                        id="trend-input-mode",
                        options=[
                            {"label": "Use editable table / draw pattern", "value": "draw"},
                            {"label": "Input matrix text", "value": "matrix"},
                        ],
                        value="draw",
                        inline=True,
                    ),
                    html.Small(
                        "Feature order is time-then-space, matching the R script. Matrix text requires one row per selected timepoint and one column per selected space.",
                        className="text-muted",
                    ),
                ], md=12),
            ],
            className="mb-2",
        ),

        dbc.Row(
            [
                dbc.Col([
                    html.Label("Editable trend table / draw the pattern"),
                    html.Div(
                        [
                            dash_table.DataTable(
                                id="trend-pattern-table",
                                data=make_pattern_table(
                                    get_ordered_window(TIME_LEVELS, DEFAULT_START_TIME, DEFAULT_END_TIME),
                                    get_ordered_window(SPACE_LEVELS, DEFAULT_START_SPACE, DEFAULT_END_SPACE),
                                ),
                                columns=[{"name": "Time", "id": "Time", "editable": False}] + [
                                    {"name": s, "id": s, "type": "numeric", "editable": True}
                                    for s in get_ordered_window(SPACE_LEVELS, DEFAULT_START_SPACE, DEFAULT_END_SPACE)
                                ],
                                editable=True,
                                style_table={"overflowX": "auto"},
                                style_cell={
                                    "textAlign": "center",
                                    "fontFamily": "Arial",
                                    "fontSize": "13px",
                                    "padding": "6px",
                                    "minWidth": "80px",
                                },
                                style_header={"fontWeight": "bold", "backgroundColor": "#f1f3f5"},
                            )
                        ],
                        id="trend-pattern-table-container",
                    ),
                    html.Small(
                        "Edit numeric values directly. Rows = selected timepoints; columns = selected spaces.",
                        className="text-muted",
                    ),
                ], md=7),
                dbc.Col([
                    html.Label("Matrix input"),
                    dcc.Textarea(
                        id="trend-matrix-text",
                        value=DEFAULT_MATRIX_TEXT,
                        style={"width": "100%", "height": "160px", "fontFamily": "monospace"},
                    ),
                    html.Small(
                        "Default matrix rows: 12h then 24h; columns: SI6, SI7, SI8, SI9, ce, co.",
                        className="text-muted",
                    ),
                ], md=5),
            ],
            className="mb-3",
        ),

        dbc.Row(
            [
                dbc.Col([
                    html.Label("Distance scaling options"),
                    dcc.Checklist(
                        id="trend-scaling-options",
                        options=[
                            {"label": "Scale each gene vector before DTW", "value": "scale_gene"},
                            {"label": "Scale trend vector before DTW", "value": "scale_trend"},
                        ],
                        value=["scale_gene", "scale_trend"],
                        inline=False,
                    ),
                ], md=4),
                dbc.Col([
                    html.Label("Top-gene curve style"),
                    dcc.Dropdown(
                        id="trend-curve-style",
                        options=[
                            {"label": "LOESS-smoothed grey curves", "value": "loess"},
                            {"label": "Connected raw feature lines", "value": "connected"},
                        ],
                        value="loess",
                        clearable=False,
                    ),
                ], md=4),
                dbc.Col([
                    html.Label("LOESS span"),
                    dbc.Input(
                        id="trend-loess-span",
                        type="number",
                        min=0.1,
                        max=1.0,
                        step=0.05,
                        value=0.8,
                    ),
                ], md=2),
                dbc.Col([
                    dbc.Button(
                        "Run DTW pattern search",
                        id="trend-run-button",
                        color="primary",
                        className="mt-4",
                    ),
                ], md=2),
            ],
            className="mb-4",
        ),

        dcc.Store(id="trend-results-store"),

        dcc.Loading(
            type="circle",
            children=[
                html.Div(id="trend-run-status"),
                dcc.Tabs(
                    id="trend-tabs",
                    value="trend3d",
                    children=[
                        dcc.Tab(label="3D Pattern", value="trend3d"),
                        dcc.Tab(label="Top Gene Curves", value="curves"),
                        dcc.Tab(label="Top Gene Heatmap", value="heatmap"),
                        dcc.Tab(label="Ranking Table", value="ranking"),
                        dcc.Tab(label="Trend Vector Table", value="trend_table"),
                        dcc.Tab(label="Feature Matrix", value="feature_matrix"),
                    ],
                ),
                html.Br(),
                html.Div(id="trend-tab-content"),
            ],
        ),

        dbc.Row(
            [
                dbc.Col([
                    dbc.Button("Download ranking table", id="trend-download-ranking-button", color="secondary", size="sm", className="mt-3 me-2"),
                    dcc.Download(id="trend-download-ranking"),
                    dbc.Button("Download trend vector", id="trend-download-vector-button", color="secondary", size="sm", className="mt-3 me-2"),
                    dcc.Download(id="trend-download-vector"),
                    dbc.Button("Download feature matrix", id="trend-download-matrix-button", color="secondary", size="sm", className="mt-3"),
                    dcc.Download(id="trend-download-matrix"),
                ], md=12),
            ],
            className="mb-5",
        ),
    ],
    fluid=True,
)


# ============================================================
# 5. Callbacks
# ============================================================

@dash.callback(
    Output("trend-pattern-table-container", "children"),
    Input("trend-start-time", "value"),
    Input("trend-end-time", "value"),
    Input("trend-start-space", "value"),
    Input("trend-end-space", "value"),
)
def update_pattern_table(start_time, end_time, start_space, end_space):
    try:
        timepoints = get_ordered_window(TIME_LEVELS, start_time, end_time)
        spacepoints = get_ordered_window(SPACE_LEVELS, start_space, end_space)
        data = make_pattern_table(timepoints, spacepoints)
        columns = [{"name": "Time", "id": "Time", "editable": False}] + [
            {"name": s, "id": s, "type": "numeric", "editable": True}
            for s in spacepoints
        ]

        return dash_table.DataTable(
            id="trend-pattern-table",
            data=data,
            columns=columns,
            editable=True,
            style_table={"overflowX": "auto"},
            style_cell={
                "textAlign": "center",
                "fontFamily": "Arial",
                "fontSize": "13px",
                "padding": "6px",
                "minWidth": "80px",
            },
            style_header={"fontWeight": "bold", "backgroundColor": "#f1f3f5"},
        )
    except Exception as e:
        return dbc.Alert(str(e), color="danger")


@dash.callback(
    Output("trend-results-store", "data"),
    Output("trend-run-status", "children"),
    Input("trend-run-button", "n_clicks"),
    State("trend-input-mode", "value"),
    State("trend-pattern-table", "data"),
    State("trend-matrix-text", "value"),
    State("trend-start-time", "value"),
    State("trend-end-time", "value"),
    State("trend-start-space", "value"),
    State("trend-end-space", "value"),
    State("trend-top-n", "value"),
    State("trend-dtw-window-fraction", "value"),
    State("trend-scaling-options", "value"),
    prevent_initial_call=False,
)
def run_trend_search(
    n_clicks,
    input_mode,
    table_data,
    matrix_text,
    start_time,
    end_time,
    start_space,
    end_space,
    top_n,
    dtw_window_fraction,
    scaling_options,
):
    try:
        timepoints = get_ordered_window(TIME_LEVELS, start_time, end_time)
        spacepoints = get_ordered_window(SPACE_LEVELS, start_space, end_space)
        scaling_options = scaling_options or []

        if input_mode == "matrix":
            trend_values = parse_matrix_text(matrix_text, timepoints, spacepoints)
        else:
            trend_values = table_data_to_values(table_data, timepoints, spacepoints)

        rank_table, trend_table, feature_matrix_out, X_for_distance, feature_order = compute_trend_search(
            timepoints=timepoints,
            spacepoints=spacepoints,
            trend_raw_values=trend_values,
            scale_gene_vector="scale_gene" in scaling_options,
            scale_trend_vector="scale_trend" in scaling_options,
            dtw_window_fraction=float(dtw_window_fraction),
        )

        payload = {
            "rank_table": rank_table.to_json(orient="split"),
            "trend_table": trend_table.to_json(orient="split"),
            "feature_matrix": feature_matrix_out.to_json(orient="split"),
            "X_for_distance": pd.DataFrame(X_for_distance, columns=feature_order).to_json(orient="split"),
            "feature_order": feature_order,
            "timepoints": timepoints,
            "spacepoints": spacepoints,
            "top_n": int(top_n),
            "scale_trend_vector": "scale_trend" in scaling_options,
        }

        top_preview = rank_table.head(5).copy()
        preview = ", ".join([f"{r.GeneName} ({r.Gene})" for r in top_preview.itertuples()])
        status = dbc.Alert(
            [
                html.Strong("DTW pattern search completed. "),
                f"{rank_table.shape[0]} genes ranked. Top genes: {preview}.",
            ],
            color="success",
            className="mt-2",
        )
        return payload, status

    except Exception as e:
        return no_update, dbc.Alert(str(e), color="danger", className="mt-2")


def _load_results(data):
    if not data:
        raise ValueError("No results available. Click 'Run DTW pattern search'.")
    rank_table = pd.read_json(io.StringIO(data["rank_table"]), orient="split")
    trend_table = pd.read_json(io.StringIO(data["trend_table"]), orient="split")
    feature_matrix = pd.read_json(io.StringIO(data["feature_matrix"]), orient="split")
    X_for_distance = pd.read_json(io.StringIO(data["X_for_distance"]), orient="split").to_numpy(dtype=float)
    return rank_table, trend_table, feature_matrix, X_for_distance


@dash.callback(
    Output("trend-tab-content", "children"),
    Input("trend-tabs", "value"),
    Input("trend-results-store", "data"),
    Input("trend-curve-style", "value"),
    Input("trend-loess-span", "value"),
)
def update_trend_tab(active_tab, data, curve_style, loess_span):
    try:
        rank_table, trend_table, feature_matrix, X_for_distance = _load_results(data)
        feature_order = data["feature_order"]
        timepoints = data["timepoints"]
        spacepoints = data["spacepoints"]
        top_n = int(data.get("top_n", DEFAULT_TOP_N))
        use_scaled = bool(data.get("scale_trend_vector", True))

        if active_tab == "trend3d":
            fig = make_trend_3d_figure(trend_table, timepoints, spacepoints, use_scaled=use_scaled)
            return dcc.Graph(figure=fig, style={"width": "100%"})

        if active_tab == "curves":
            fig = make_top_gene_curve_figure(
                rank_table=rank_table,
                trend_table=trend_table,
                X_for_distance=X_for_distance,
                feature_order=feature_order,
                top_n=top_n,
                curve_style=curve_style,
                loess_span=float(loess_span),
            )
            return dcc.Graph(figure=fig, style={"width": "100%"})

        if active_tab == "heatmap":
            # Heatmap intentionally uses raw Beta/logFC from feature_matrix,
            # not X_for_distance, because X_for_distance may be z-scored for DTW.
            fig = make_top_heatmap_figure(rank_table, feature_matrix, feature_order, top_n=top_n)
            return dcc.Graph(figure=fig, style={"width": "100%"})

        if active_tab == "trend_table":
            df = trend_table.copy()
        elif active_tab == "feature_matrix":
            df = feature_matrix.copy()
        else:
            df = rank_table.copy()

        for col in df.select_dtypes(include=[np.number]).columns:
            df[col] = df[col].round(4)

        return dash_table.DataTable(
            data=df.to_dict("records"),
            columns=[{"name": c, "id": c} for c in df.columns],
            page_size=20,
            filter_action="native",
            sort_action="native",
            style_table={"overflowX": "auto"},
            style_cell={
                "textAlign": "left",
                "fontFamily": "Arial",
                "fontSize": "13px",
                "padding": "6px",
                "minWidth": "110px",
                "whiteSpace": "normal",
            },
            style_header={"fontWeight": "bold", "backgroundColor": "#f1f3f5"},
        )
    except Exception as e:
        return dbc.Alert(str(e), color="warning")


@dash.callback(
    Output("trend-download-ranking", "data"),
    Input("trend-download-ranking-button", "n_clicks"),
    State("trend-results-store", "data"),
    prevent_initial_call=True,
)
def download_ranking(n_clicks, data):
    if not n_clicks:
        return no_update
    rank_table, _, _, _ = _load_results(data)
    return dcc.send_data_frame(rank_table.to_csv, "all_genes_rank_by_DTW_to_predefined_trend.csv", index=False)


@dash.callback(
    Output("trend-download-vector", "data"),
    Input("trend-download-vector-button", "n_clicks"),
    State("trend-results-store", "data"),
    prevent_initial_call=True,
)
def download_vector(n_clicks, data):
    if not n_clicks:
        return no_update
    _, trend_table, _, _ = _load_results(data)
    return dcc.send_data_frame(trend_table.to_csv, "predefined_trend_vector.csv", index=False)


@dash.callback(
    Output("trend-download-matrix", "data"),
    Input("trend-download-matrix-button", "n_clicks"),
    State("trend-results-store", "data"),
    prevent_initial_call=True,
)
def download_matrix(n_clicks, data):
    if not n_clicks:
        return no_update
    _, _, feature_matrix, _ = _load_results(data)
    return dcc.send_data_frame(feature_matrix.to_csv, "feature_matrix_used_for_DTW_to_predefined_trend.csv", index=False)
