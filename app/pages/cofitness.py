import os
from functools import lru_cache

import numpy as np
import pandas as pd
import dash
from dash import html, dcc, dash_table, Input, Output, State, no_update
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist


dash.register_page(__name__, path="/cofitness", name="Cofitness")


# ============================================================
# 0. Paths
# ============================================================

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(APP_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

COFIT_DIR = os.path.join(DATA_DIR, "cofitness")

SPEARMAN_FILE = os.path.join(
    COFIT_DIR,
    "cofitness_correlation_matrix_spearman.csv.gz"
)

PEARSON_FILE = os.path.join(
    COFIT_DIR,
    "cofitness_correlation_matrix_pearson.csv.gz"
)

OLD_SINGLE_MATRIX_FILE = os.path.join(
    COFIT_DIR,
    "cofitness_correlation_matrix.csv"
)

ANNOTATION_FILE = os.path.join(
    DATA_DIR,
    "annotation",
    "new_annotations_with_uniprot_names.csv"
)

DEFAULT_HEATMAP_N_TOP = 20
AUTO_HIDE_LABEL_THRESHOLD = 30
MAX_DISPLAY_TABLE_DIM = 120
MAX_SINGLE_GENE_TOP_N = 100
MAX_HEATMAP_N_GENES = 1500
PUBLISHABLE_HEATMAP_COLORSCALE = [
    [0.0, "#2166AC"],
    [0.5, "#FFFFFF"],
    [1.0, "#B2182B"],
]



# ============================================================
# 1. Data loading
# ============================================================

def read_cor_matrix(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Correlation matrix file not found: {path}")

    df = pd.read_csv(path)

    if "Gene" not in df.columns:
        raise ValueError(f"The first column must be named 'Gene' in file: {path}")

    gene_ids = df["Gene"].astype(str).tolist()

    mat_df = df.drop(columns=["Gene"]).copy()
    mat_df = mat_df.apply(pd.to_numeric, errors="coerce")

    mat = pd.DataFrame(
        mat_df.values,
        index=gene_ids,
        columns=mat_df.columns.astype(str)
    )

    return mat


@lru_cache(maxsize=1)
def load_correlation_matrices():
    spearman_path = SPEARMAN_FILE

    if not os.path.exists(spearman_path) and os.path.exists(OLD_SINGLE_MATRIX_FILE):
        spearman_path = OLD_SINGLE_MATRIX_FILE

    spearman = read_cor_matrix(spearman_path)

    if os.path.exists(PEARSON_FILE):
        pearson = read_cor_matrix(PEARSON_FILE)
    else:
        pearson = spearman.copy()

    common_genes = sorted(
        set(spearman.index)
        & set(spearman.columns)
        & set(pearson.index)
        & set(pearson.columns)
    )

    if len(common_genes) == 0:
        raise ValueError("No common genes found between Spearman and Pearson matrices.")

    spearman = spearman.loc[common_genes, common_genes]
    pearson = pearson.loc[common_genes, common_genes]

    return {
        "spearman": spearman,
        "pearson": pearson,
    }


@lru_cache(maxsize=1)
def load_annotation():
    if not os.path.exists(ANNOTATION_FILE):
        mats = load_correlation_matrices()
        genes = list(mats["spearman"].index)
        return pd.DataFrame({"Gene": genes, "GeneName": genes})

    ann = pd.read_csv(ANNOTATION_FILE)

    if "locus_ID" not in ann.columns:
        mats = load_correlation_matrices()
        genes = list(mats["spearman"].index)
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

    ann = ann.rename(columns={"locus_ID": "Gene"})
    ann = ann[["Gene", "GeneName"]].drop_duplicates()

    return ann


@lru_cache(maxsize=1)
def load_gene_lookup():
    mats = load_correlation_matrices()
    genes = list(mats["spearman"].index)

    ann = load_annotation()

    lookup = pd.DataFrame({"Gene": genes})
    lookup = lookup.merge(ann, on="Gene", how="left")
    lookup["GeneName"] = lookup["GeneName"].fillna(lookup["Gene"])
    lookup["DisplayLabel"] = lookup["GeneName"] + " | " + lookup["Gene"]

    lookup = lookup.sort_values(["GeneName", "Gene"]).reset_index(drop=True)

    gene_to_name = dict(zip(lookup["Gene"], lookup["GeneName"]))
    gene_to_label = dict(zip(lookup["Gene"], lookup["DisplayLabel"]))

    dropdown_options = [
        {"label": row["DisplayLabel"], "value": row["Gene"]}
        for _, row in lookup.iterrows()
    ]

    return lookup, gene_to_name, gene_to_label, dropdown_options


def display_gene_name(gene_id):
    _, gene_to_name, _, _ = load_gene_lookup()
    return gene_to_name.get(gene_id, gene_id)


def display_gene_label(gene_id):
    _, _, gene_to_label, _ = load_gene_lookup()
    return gene_to_label.get(gene_id, gene_id)


def get_active_matrix(cor_method):
    mats = load_correlation_matrices()
    if cor_method not in mats:
        cor_method = "spearman"
    return mats[cor_method]


def get_cor_label(cor_method):
    return "Spearman" if cor_method == "spearman" else "Pearson"


def get_gene_count():
    mats = load_correlation_matrices()
    return mats["spearman"].shape[0]


# ============================================================
# 2. Heatmap helpers
# ============================================================

def safe_int(value, default=DEFAULT_HEATMAP_N_TOP, minimum=1, maximum=None):
    """Convert Dash numeric input to a safe integer.

    Dash can temporarily pass None when a number box is being edited, especially
    when the value exceeds the HTML input max. This helper prevents the
    int(None) error and clamps the requested value to the matrix size.
    """
    try:
        if value is None or value == "":
            value = default
        value = int(float(value))
    except (TypeError, ValueError):
        value = default

    value = max(minimum, value)
    if maximum is not None:
        value = min(value, maximum)

    return value


def get_top_variable_genes(mat, n_top):
    gene_sd = mat.std(axis=1, skipna=True)
    n = safe_int(n_top, default=DEFAULT_HEATMAP_N_TOP, minimum=1, maximum=len(gene_sd))
    return gene_sd.sort_values(ascending=False).head(n).index.tolist()


def get_bottom_variable_genes(mat, n_bottom):
    gene_sd = mat.std(axis=1, skipna=True)
    n = safe_int(n_bottom, default=DEFAULT_HEATMAP_N_TOP, minimum=1, maximum=len(gene_sd))
    return gene_sd.sort_values(ascending=True).head(n).index.tolist()


def get_heatmap_gene_list(mat, heatmap_gene_mode, n_top):
    total_genes = mat.shape[0]

    effective_n = safe_int(
        n_top,
        default=DEFAULT_HEATMAP_N_TOP,
        minimum=1,
        maximum=min(MAX_HEATMAP_N_GENES, total_genes),
    )

    if heatmap_gene_mode == "bottom":
        selected_genes = get_bottom_variable_genes(mat, effective_n)
        return selected_genes, f"bottom {len(selected_genes)} variable genes", effective_n

    # Default to top variable genes. This also protects against any stale browser state
    # that might still send a removed/old value such as "all".
    selected_genes = get_top_variable_genes(mat, effective_n)
    return selected_genes, f"top {len(selected_genes)} variable genes", effective_n


def cluster_matrix(mat):
    if mat.shape[0] <= 2:
        return mat

    row_dist = pdist(mat.fillna(0).values)
    col_dist = pdist(mat.fillna(0).T.values)

    row_linkage = linkage(row_dist, method="complete")
    col_linkage = linkage(col_dist, method="complete")

    row_order = leaves_list(row_linkage)
    col_order = leaves_list(col_linkage)

    return mat.iloc[row_order, col_order]


def get_heatmap_limits(mat, scale_method):
    vals = mat.values.flatten()
    vals = vals[np.isfinite(vals)]

    if len(vals) == 0:
        return -1, 1, "No finite values"

    if scale_method == "fixed":
        return -1, 1, "Fixed range: [-1, 1]"

    if scale_method == "quantile":
        zmin = float(np.nanquantile(vals, 0.05))
        zmax = float(np.nanquantile(vals, 0.95))

        if zmin == zmax:
            zmin = float(np.nanmin(vals))
            zmax = float(np.nanmax(vals))

        if zmin == zmax:
            zmin -= 0.01
            zmax += 0.01

        return zmin, zmax, f"Quantile range 5%-95%: [{zmin:.3f}, {zmax:.3f}]"

    zmin = float(np.nanmin(vals))
    zmax = float(np.nanmax(vals))

    if zmin == zmax:
        zmin -= 0.01
        zmax += 0.01

    return zmin, zmax, f"Dynamic range: [{zmin:.3f}, {zmax:.3f}]"


def should_show_axis_labels(label_mode, n_genes):
    if label_mode == "show":
        return True
    if label_mode == "hide":
        return False
    return n_genes <= AUTO_HIDE_LABEL_THRESHOLD


def make_heatmap_axis_values_and_ticks(genes, label_mode, axis_name):
    n_genes = len(genes)
    show_labels = should_show_axis_labels(label_mode, n_genes)
    numeric_values = list(range(n_genes))

    if show_labels:
        tickvals = numeric_values
        ticktext = [display_gene_label(g) for g in genes]
    else:
        tickvals = []
        ticktext = []

    title_suffix = "gene labels shown" if show_labels else "gene labels hidden"
    axis_title = f"{axis_name} gene ({title_suffix})"

    return numeric_values, tickvals, ticktext, show_labels, axis_title


def estimate_heatmap_margins(n_genes, show_row_labels, show_col_labels, row_genes, col_genes):
    """Return compact margins/font sizes for a publication-style heatmap."""
    if n_genes <= 30:
        tick_font_size = 8
    elif n_genes <= 80:
        tick_font_size = 6
    elif n_genes <= 150:
        tick_font_size = 5
    else:
        tick_font_size = 4

    if show_row_labels:
        row_label_lengths = [len(display_gene_label(g)) for g in row_genes]
        max_row_len = max(row_label_lengths) if row_label_lengths else 12
        left_margin = min(210, max(105, int(max_row_len * 4.8)))
    else:
        left_margin = 45

    if show_col_labels:
        col_label_lengths = [len(display_gene_label(g)) for g in col_genes]
        max_col_len = max(col_label_lengths) if col_label_lengths else 12
        bottom_margin = min(230, max(115, int(max_col_len * 5.0)))
    else:
        bottom_margin = 55

    right_margin = 70
    top_margin = 92

    return left_margin, right_margin, top_margin, bottom_margin, tick_font_size


def make_heatmap_customdata(row_genes, col_genes):
    customdata = []
    for row_gene in row_genes:
        row_name = display_gene_name(row_gene)
        row_label = display_gene_label(row_gene)
        row_items = []
        for col_gene in col_genes:
            col_name = display_gene_name(col_gene)
            col_label = display_gene_label(col_gene)
            row_items.append([
                row_label,
                row_gene,
                row_name,
                col_label,
                col_gene,
                col_name,
            ])
        customdata.append(row_items)

    return np.array(customdata, dtype=object)


def make_heatmap_figure(
    cor_method,
    heatmap_gene_mode,
    n_top_genes,
    cluster_genes,
    scale_method,
    row_label_mode,
    col_label_mode,
):
    mat = get_active_matrix(cor_method)
    cor_label = get_cor_label(cor_method)
    total_genes = mat.shape[0]

    selected_genes, gene_selection_label, effective_n_top = get_heatmap_gene_list(
        mat,
        heatmap_gene_mode=heatmap_gene_mode,
        n_top=n_top_genes,
    )

    # This line is the core fix for the variable-gene option:
    # when the user enters 100, the selected matrix is explicitly rebuilt as
    # 100 x 100 before clustering or plotting.
    sub = mat.loc[selected_genes, selected_genes].copy()

    raw_min = float(np.nanmin(sub.values))
    raw_max = float(np.nanmax(sub.values))

    if cluster_genes == "yes":
        sub = cluster_matrix(sub)

    zmin, zmax, scale_label = get_heatmap_limits(sub, scale_method)

    row_genes = sub.index.astype(str).tolist()
    col_genes = sub.columns.astype(str).tolist()
    n_display = sub.shape[0]

    x_values, x_tickvals, x_ticktext, show_col_labels, x_title = make_heatmap_axis_values_and_ticks(
        col_genes,
        col_label_mode,
        "Column",
    )
    y_values, y_tickvals, y_ticktext, show_row_labels, y_title = make_heatmap_axis_values_and_ticks(
        row_genes,
        row_label_mode,
        "Row",
    )

    customdata = make_heatmap_customdata(row_genes, col_genes)

    # Compact, publication-style layout. Axis titles are removed and margins
    # are kept tight so row labels sit close to the heatmap instead of floating
    # far to the left.
    left_margin, right_margin, top_margin, bottom_margin, tick_font_size = estimate_heatmap_margins(
        n_display,
        show_row_labels,
        show_col_labels,
        row_genes,
        col_genes,
    )

    if n_display <= 30:
        heatmap_height = max(640, 18 * n_display + 250)
    elif n_display <= 120:
        heatmap_height = 820
    elif n_display <= 500:
        heatmap_height = 920
    else:
        heatmap_height = 1020

    fig = go.Figure(
        data=go.Heatmap(
            x=x_values,
            y=y_values,
            z=sub.values,
            customdata=customdata,
            colorscale=PUBLISHABLE_HEATMAP_COLORSCALE,
            zmin=zmin,
            zmax=zmax,
            zauto=False,
            colorbar=dict(
                title=dict(text=f"{cor_label}<br>correlation", side="right"),
                tickformat=".2f",
                thickness=12,
                len=0.76,
                x=1.005,
                xanchor="left",
                y=0.50,
            ),
            xgap=0.2 if n_display <= 120 else 0,
            ygap=0.2 if n_display <= 120 else 0,
            hovertemplate=(
                "<b>Row gene</b>: %{customdata[0]}<br>"
                "Row Gene ID: %{customdata[1]}<br>"
                "Row GeneName: %{customdata[2]}<br><br>"
                "<b>Column gene</b>: %{customdata[3]}<br>"
                "Column Gene ID: %{customdata[4]}<br>"
                "Column GeneName: %{customdata[5]}<br><br>"
                f"{cor_label} correlation: "
                "%{z:.3f}<extra></extra>"
            )
        )
    )

    fig.update_layout(
        title=dict(
            text=(
                f"{cor_label} Co-fitness Correlation Matrix<br>"
                f"<sup>{gene_selection_label}; displayed {n_display} of {total_genes} genes | "
                f"Raw range: [{raw_min:.3f}, {raw_max:.3f}] | "
                f"{scale_label}</sup>"
            ),
            font=dict(size=18),
            x=0.01,
            xanchor="left",
        ),
        template="plotly_white",
        height=heatmap_height,
        margin=dict(l=left_margin, r=right_margin, t=top_margin, b=bottom_margin),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="Arial", size=12, color="#222222"),
        xaxis=dict(
            title="",
            tickmode="array",
            tickvals=x_tickvals,
            ticktext=x_ticktext,
            tickangle=-45,
            tickfont=dict(size=tick_font_size),
            showticklabels=show_col_labels,
            showgrid=False,
            zeroline=False,
            ticks="",
            automargin=False,
            constrain="domain",
        ),
        yaxis=dict(
            title="",
            tickmode="array",
            tickvals=y_tickvals,
            ticktext=y_ticktext,
            tickfont=dict(size=tick_font_size),
            showticklabels=show_row_labels,
            showgrid=False,
            zeroline=False,
            autorange="reversed",
            ticks="",
            automargin=False,
            constrain="domain",
        ),
    )

    label_note = (
        "Gene labels are shown."
        if show_row_labels or show_col_labels
        else f"Gene labels are hidden because the selected matrix contains >{AUTO_HIDE_LABEL_THRESHOLD} genes or labels were manually hidden. Hover over cells to see full GeneName | Gene ID."
    )

    metadata = {
        "sub": sub,
        "n_display": n_display,
        "total_genes": total_genes,
        "gene_selection_label": gene_selection_label,
        "effective_n_top": effective_n_top,
        "show_row_labels": show_row_labels,
        "show_col_labels": show_col_labels,
        "label_note": label_note,
        "scale_label": scale_label,
    }

    return fig, sub, metadata


# ============================================================
# 3. Single-gene cofitness helpers
# ============================================================

def get_single_gene_correlation_table(cor_method, gene_id, top_n, direction):
    mat = get_active_matrix(cor_method)
    cor_label = get_cor_label(cor_method)

    if gene_id not in mat.index:
        raise ValueError(f"Gene not found in correlation matrix: {gene_id}")

    top_n = safe_int(top_n, default=20, minimum=1, maximum=min(MAX_SINGLE_GENE_TOP_N, mat.shape[0] - 1))

    v = mat.loc[gene_id].copy()
    v = v.drop(labels=[gene_id], errors="ignore")
    v = v.replace([np.inf, -np.inf], np.nan).dropna()

    if direction == "positive":
        v = v[v > 0].sort_values(ascending=False)
    elif direction == "negative":
        v = v[v < 0].sort_values(ascending=True)
    else:
        v = v[v.abs() > 0]
        v = v.loc[v.abs().sort_values(ascending=False).index]

    v = v.head(top_n)

    df = pd.DataFrame({
        "Rank": range(1, len(v) + 1),
        "Partner_GeneID": v.index.astype(str),
        "Partner_GeneName": [display_gene_name(g) for g in v.index],
        "Correlation_Method": cor_label,
        "Correlation": v.values,
        "Direction": np.where(v.values > 0, "Positive", "Negative"),
        "Abs_Correlation": np.abs(v.values),
    })

    df["Partner_Label"] = df["Partner_GeneName"] + " | " + df["Partner_GeneID"]

    return df


def make_single_gene_figure(cor_method, gene_id, top_n, direction):
    df = get_single_gene_correlation_table(
        cor_method=cor_method,
        gene_id=gene_id,
        top_n=top_n,
        direction=direction
    )

    cor_label = get_cor_label(cor_method)
    query_label = display_gene_label(gene_id)

    df_plot = df.sort_values("Correlation", ascending=True).copy()

    title_direction = {
        "positive": "Positive",
        "negative": "Negative",
        "absolute": "Absolute",
    }.get(direction, direction)

    marker_colors = np.where(df_plot["Correlation"] > 0, "darkred", "darkblue")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df_plot["Correlation"],
            y=df_plot["Partner_Label"],
            mode="markers",
            marker=dict(size=10, color=marker_colors),
            customdata=np.stack(
                [
                    df_plot["Partner_GeneName"],
                    df_plot["Partner_GeneID"],
                    df_plot["Abs_Correlation"].round(3),
                ],
                axis=-1
            ),
            hovertemplate=(
                "Partner GeneName: %{customdata[0]}<br>"
                "Partner Gene ID: %{customdata[1]}<br>"
                f"{cor_label} correlation: "
                "%{x:.3f}<br>"
                "Absolute correlation: %{customdata[2]:.3f}"
                "<extra></extra>"
            ),
        )
    )

    # Show the zero-reference line only for the Top absolute mode.
    # For Top positive or Top negative mode, all selected points are on one side of zero,
    # so the zero line can visually dominate the plot and is intentionally hidden.
    if direction == "absolute":
        fig.add_vline(
            x=0,
            line_dash="dash",
            line_color="gray"
        )

    fig.update_layout(
        title=(
            f"{cor_label} co-fitness of {query_label} "
            f"(Top {len(df_plot)} {title_direction} Correlations)"
        ),
        template="plotly_white",
        height=max(520, 28 * len(df_plot) + 180),
        margin=dict(l=240, r=60, t=90, b=70),
        xaxis_title=f"{cor_label} correlation",
        yaxis_title="",
        yaxis=dict(categoryorder="array", categoryarray=df_plot["Partner_Label"].tolist()),
    )

    return fig, df


# ============================================================
# 4. Summary stats
# ============================================================

def make_summary_stats(cor_method):
    mat = get_active_matrix(cor_method)
    cor_label = get_cor_label(cor_method)

    vals = mat.values.flatten()
    vals = vals[np.isfinite(vals)]

    summary_text = (
        f"{cor_label} Co-fitness Correlation Matrix Summary\n"
        f"======================================\n\n"
        f"Matrix dimensions: {mat.shape[0]} x {mat.shape[1]}\n"
        f"Number of genes: {mat.shape[0]}\n\n"
        f"Correlation range: {np.nanmin(vals):.3f} to {np.nanmax(vals):.3f}\n"
        f"Mean correlation: {np.nanmean(vals):.3f}\n"
        f"Median correlation: {np.nanmedian(vals):.3f}"
    )

    return summary_text


# ============================================================
# 5. Annotation text blocks
# ============================================================

def page_method_annotation():
    return dbc.Alert(
        [
            html.Strong("Important interpretation note: "),
            html.Span(
                "Spearman and Pearson co-fitness correlations measure pairwise association between two gene fitness profiles. "
                "They are useful for finding genes with similar or opposite fitness patterns, but they do not distinguish "
                "direct associations from indirect associations. For example, two genes can be correlated because both are "
                "associated with a third gene or pathway. To further separate direct from indirect relationships, use the "
            ),
            dcc.Link("Network Browser page", href="/network-browser"),
            html.Span(
                ", which is designed for network-level analysis such as partial-correlation or graphical-model-based relationships."
            ),
        ],
        color="warning",
        className="mb-4",
    )


def heatmap_module_annotation():
    return dbc.Alert(
        [
            html.Strong("How to use this heatmap: "),
            html.Span(
                "The heatmap shows pairwise co-fitness correlations among selected genes. "
                "Top variable genes are ranked by SD_across_correlations, meaning genes with the largest standard "
                "deviation across their correlations with all other genes. Bottom variable genes have the smallest "
                "SD_across_correlations and therefore more uniform or less contrasting co-fitness profiles. "
                "The interactive heatmap is capped at 1,500 genes to keep browser rendering responsive; "
                "to analyze or plot the full matrix, download the selected matrix table and plot it locally."
            ),
            html.Br(),
            html.Span(
                f"Row and column GeneName labels are set to 'Auto' by default. In Auto mode, labels are shown only when "
                f"the displayed matrix has {AUTO_HIDE_LABEL_THRESHOLD} genes or fewer; for larger matrices they are hidden "
                "to avoid a messy plot. Even when labels are hidden, hovering over any cell shows the full GeneName and Gene ID."
            ),
            html.Br(),
            html.Span(
                "Values above 1,500 are capped at 1,500 to keep the heatmap readable and responsive. "
                "To plot the full 3,330 genes, download the table and plot the full matrix locally."
            ),
        ],
        color="secondary",
        className="mb-3",
    )


def heatmap_table_annotation(metadata=None):
    metadata = metadata or {}
    label_note = metadata.get("label_note", "Hover over heatmap cells to see full gene labels.")

    return dbc.Alert(
        [
            html.Strong("About this co-fitness matrix table: "),
            html.Span(
                "Each row and column represents a gene. Each cell shows the selected co-fitness correlation value "
                "between the row gene and the column gene. Positive values indicate similar fitness behavior across "
                "conditions, while negative values indicate opposite fitness behavior. Values close to 0 indicate weak "
                "or no pairwise co-fitness relationship."
            ),
            html.Br(),
            html.Span(label_note),
        ],
        color="info",
        className="mb-3",
    )


def single_gene_table_annotation():
    return dbc.Alert(
        [
            html.Strong("About this single-gene co-fitness table: "),
            html.Span(
                "This table lists the genes most strongly co-fit with the selected query gene. "
                "The correlation value measures how similarly two genes behave across the underlying "
                "fitness profiles or experimental conditions."
            ),
            html.Br(),
            html.Span(
                "Positive correlations suggest that two genes tend to have similar fitness patterns. "
                "Negative correlations suggest opposite fitness patterns. The absolute correlation "
                "column ranks genes by the strength of the relationship regardless of sign."
            ),
        ],
        color="info",
        className="mb-3",
    )


def summary_annotation():
    return dbc.Alert(
        [
            html.Strong("About these summary statistics: "),
            html.Span(
                "This section summarizes the global structure of the selected co-fitness "
                "correlation matrix. The table highlights genes with the largest standard "
                "deviation across their correlations with all other genes."
            ),
            html.Br(),
            html.Span(
                "A high SD_across_correlations means that a gene has a highly variable "
                "co-fitness profile: it is strongly correlated with some genes but weakly "
                "or negatively correlated with others. These genes often produce stronger "
                "contrast in the heatmap."
            ),
        ],
        color="info",
        className="mb-3",
    )


# ============================================================
# 6. Layout defaults
# ============================================================

try:
    _, _, _, gene_dropdown_options = load_gene_lookup()
    default_gene = gene_dropdown_options[0]["value"] if gene_dropdown_options else None
    total_gene_count = get_gene_count()
except Exception:
    gene_dropdown_options = []
    default_gene = None
    total_gene_count = None


# ============================================================
# 7. Layout
# ============================================================

layout = dbc.Container(
    [
        html.H2("Cofitness", className="page-title"),

        html.P(
            "Explore pairwise co-fitness correlations between genes using Spearman or Pearson correlation matrices. "
            "The single-gene co-fitness plot is shown first, followed by the interactive co-fitness heatmap.",
            className="lead"
        ),

        page_method_annotation(),

        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Label("Correlation method"),
                        dcc.RadioItems(
                            id="cofit-cor-method",
                            options=[
                                {"label": "Spearman", "value": "spearman"},
                                {"label": "Pearson", "value": "pearson"},
                            ],
                            value="spearman",
                            inline=True,
                        ),
                    ],
                    md=6,
                ),
            ],
            className="mb-4",
        ),

        html.Hr(),

        # ======================================================
        # PART 1: Single gene co-fitness plot
        # ======================================================

        html.H3("1. Single Gene Co-fitness Plot"),

        dbc.Alert(
            [
                html.Strong("How to use this module: "),
                html.Span(
                    "Select a query gene to identify genes with the strongest positive, negative, "
                    "or absolute pairwise co-fitness correlations. Positive correlations indicate similar "
                    "fitness profiles, while negative correlations indicate opposite fitness profiles. "
                    "These correlations are pairwise associations and should not be interpreted as direct regulation. "
                    f"For readability, the number of top correlated partner genes is limited to {MAX_SINGLE_GENE_TOP_N}."
                ),
            ],
            color="secondary",
            className="mb-3",
        ),

        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Label("Search by Gene ID or Gene Name"),
                        dcc.Dropdown(
                            id="cofit-gene-select",
                            options=gene_dropdown_options,
                            value=default_gene,
                            searchable=True,
                            clearable=False,
                        ),
                    ],
                    md=5,
                ),
                dbc.Col(
                    [
                        html.Label("Number of top correlated genes"),
                        dcc.Input(
                            id="cofit-top-n",
                            type="number",
                            min=1,
                            max=MAX_SINGLE_GENE_TOP_N,
                            step=1,
                            value=20,
                            debounce=False,
                            className="form-control",
                            style={"width": "100%"},
                        ),
                        html.Small(
                            f"Choose 1–{MAX_SINGLE_GENE_TOP_N} partner genes. Values above {MAX_SINGLE_GENE_TOP_N} are capped at {MAX_SINGLE_GENE_TOP_N} to keep the single-gene plot readable and responsive. Press Enter or click outside the box if your browser does not update immediately.",
                            className="text-muted",
                        ),
                    ],
                    md=3,
                ),
                dbc.Col(
                    [
                        html.Label("Correlation direction"),
                        dcc.Dropdown(
                            id="cofit-correlation-direction",
                            options=[
                                {"label": "Top positive", "value": "positive"},
                                {"label": "Top negative", "value": "negative"},
                                {"label": "Top absolute", "value": "absolute"},
                            ],
                            value="positive",
                            clearable=False,
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
                        dbc.Button(
                            "Download single-gene table",
                            id="cofit-download-single-table-button",
                            color="secondary",
                            size="sm",
                            className="mb-2",
                        ),
                        dcc.Download(id="cofit-download-single-table"),
                    ],
                    md=4,
                ),
            ],
            className="mb-3",
        ),

        dcc.Loading(
            type="circle",
            children=[
                dcc.Graph(id="cofit-single-gene-plot", style={"width": "100%"}),
                html.Div(id="cofit-single-gene-table"),
            ],
        ),

        html.Hr(),

        # ======================================================
        # PART 2: Interactive heatmap
        # ======================================================

        html.H3("2. Interactive Co-fitness Heatmap"),

        heatmap_module_annotation(),

        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Label("Gene selection mode"),
                        dcc.RadioItems(
                            id="cofit-heatmap-gene-mode",
                            options=[
                                {"label": "Top variable genes", "value": "top"},
                                {"label": "Bottom variable genes", "value": "bottom"},
                            ],
                            value="top",
                            inline=True,
                            inputStyle={"marginRight": "6px", "marginLeft": "10px"},
                        ),
                        html.Small(
                            "Use top or bottom variable genes for ranked subsets of the co-fitness matrix.",
                            className="text-muted",
                        ),
                    ],
                    md=4,
                ),
                dbc.Col(
                    [
                        html.Label("Number of top/bottom variable genes"),
                        dcc.Input(
                            id="cofit-n-top-genes",
                            type="number",
                            min=1,
                            max=MAX_HEATMAP_N_GENES,
                            step=1,
                            value=DEFAULT_HEATMAP_N_TOP,
                            debounce=False,
                            className="form-control",
                            style={"width": "100%"},
                        ),
                        html.Small(
                            f"Top variable genes are ranked by SD_across_correlations. Bottom variable genes have the smallest SD_across_correlations. Available genes: {total_gene_count if total_gene_count is not None else 'unknown'}. Values above 1,500 are capped at 1,500 to keep the heatmap readable and responsive. To plot the full 3,330 genes, download the table and plot the full matrix locally. Press Enter or click outside the box if your browser does not update immediately.",
                            className="text-muted",
                        ),
                    ],
                    md=3,
                ),
                dbc.Col(
                    [
                        html.Label("Cluster genes"),
                        dcc.RadioItems(
                            id="cofit-cluster-genes",
                            options=[
                                {"label": "Yes", "value": "yes"},
                                {"label": "No", "value": "no"},
                            ],
                            value="yes",
                            inline=True,
                            inputStyle={"marginRight": "6px", "marginLeft": "10px"},
                        ),
                    ],
                    md=2,
                ),
                dbc.Col(
                    [
                        html.Label("Heatmap color scaling"),
                        dcc.Dropdown(
                            id="cofit-heatmap-scale",
                            options=[
                                {
                                    "label": "Dynamic min-max in selected genes",
                                    "value": "dynamic",
                                },
                                {
                                    "label": "Quantile scaling 5%-95%",
                                    "value": "quantile",
                                },
                                {
                                    "label": "Fixed full correlation range -1 to 1",
                                    "value": "fixed",
                                },
                            ],
                            value="dynamic",
                            clearable=False,
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
                        html.Label("Row gene labels"),
                        dcc.Dropdown(
                            id="cofit-row-label-mode",
                            options=[
                                {"label": f"Auto: show only if ≤ {AUTO_HIDE_LABEL_THRESHOLD} genes", "value": "auto"},
                                {"label": "Always show", "value": "show"},
                                {"label": "Always hide", "value": "hide"},
                            ],
                            value="auto",
                            clearable=False,
                        ),
                    ],
                    md=4,
                ),
                dbc.Col(
                    [
                        html.Label("Column gene labels"),
                        dcc.Dropdown(
                            id="cofit-column-label-mode",
                            options=[
                                {"label": f"Auto: show only if ≤ {AUTO_HIDE_LABEL_THRESHOLD} genes", "value": "auto"},
                                {"label": "Always show", "value": "show"},
                                {"label": "Always hide", "value": "hide"},
                            ],
                            value="auto",
                            clearable=False,
                        ),
                    ],
                    md=4,
                ),
                dbc.Col(
                    [
                        html.Label("Download matrix"),
                        html.Br(),
                        dbc.Button(
                            "Download heatmap matrix table",
                            id="cofit-download-heatmap-table-button",
                            color="secondary",
                            size="sm",
                            className="mb-2",
                        ),
                        dcc.Download(id="cofit-download-heatmap-table"),
                    ],
                    md=4,
                ),
            ],
            className="mb-3",
        ),

        dcc.Loading(
            type="circle",
            children=[
                html.Div(id="cofit-heatmap-status"),
                dcc.Graph(id="cofit-heatmap-plot", style={"width": "100%"}),
                html.Div(id="cofit-heatmap-table"),
            ],
        ),

        html.Hr(),

        # ======================================================
        # PART 3: Summary
        # ======================================================

        html.H3("3. Matrix Summary Statistics"),

        dcc.Loading(
            type="circle",
            children=[
                html.Div(id="cofit-summary-stats")
            ],
        ),
    ],
    fluid=True
)


# ============================================================
# 8. Callbacks
# ============================================================

@dash.callback(
    Output("cofit-single-gene-plot", "figure"),
    Output("cofit-single-gene-table", "children"),
    Input("cofit-cor-method", "value"),
    Input("cofit-gene-select", "value"),
    Input("cofit-top-n", "value"),
    Input("cofit-top-n", "n_submit"),
    Input("cofit-top-n", "n_blur"),
    Input("cofit-correlation-direction", "value"),
)
def update_single_gene_section(
    cor_method,
    gene_select,
    top_n,
    top_n_submit,
    top_n_blur,
    correlation_direction,
):
    try:
        if gene_select is None:
            fig = go.Figure()
            fig.update_layout(
                template="plotly_white",
                title="No gene selected"
            )
            return fig, dbc.Alert("No gene selected.", color="warning")

        fig, df = make_single_gene_figure(
            cor_method=cor_method,
            gene_id=gene_select,
            top_n=top_n,
            direction=correlation_direction,
        )

        df_show = df.copy()
        df_show["Correlation"] = df_show["Correlation"].round(3)
        df_show["Abs_Correlation"] = df_show["Abs_Correlation"].round(3)

        keep_cols = [
            "Rank",
            "Partner_GeneID",
            "Partner_GeneName",
            "Correlation_Method",
            "Correlation",
            "Direction",
            "Abs_Correlation",
        ]

        df_show = df_show[keep_cols]

        table = html.Div(
            [
                single_gene_table_annotation(),

                dash_table.DataTable(
                    data=df_show.to_dict("records"),
                    columns=[{"name": c, "id": c} for c in df_show.columns],
                    page_size=15,
                    filter_action="native",
                    sort_action="native",
                    style_table={"overflowX": "auto"},
                    style_cell={
                        "textAlign": "left",
                        "fontFamily": "Arial",
                        "fontSize": "14px",
                        "padding": "6px",
                        "minWidth": "120px",
                        "whiteSpace": "normal",
                    },
                    style_header={
                        "fontWeight": "bold",
                        "backgroundColor": "#f1f3f5",
                    },
                ),
            ]
        )

        return fig, table

    except Exception as e:
        fig = go.Figure()
        fig.update_layout(
            template="plotly_white",
            title="Error",
            annotations=[
                dict(
                    text=str(e),
                    x=0.5,
                    y=0.5,
                    showarrow=False
                )
            ]
        )
        return fig, dbc.Alert(str(e), color="danger")


@dash.callback(
    Output("cofit-heatmap-plot", "figure"),
    Output("cofit-heatmap-table", "children"),
    Output("cofit-heatmap-status", "children"),
    Input("cofit-cor-method", "value"),
    Input("cofit-heatmap-gene-mode", "value"),
    Input("cofit-n-top-genes", "value"),
    Input("cofit-n-top-genes", "n_submit"),
    Input("cofit-n-top-genes", "n_blur"),
    Input("cofit-cluster-genes", "value"),
    Input("cofit-heatmap-scale", "value"),
    Input("cofit-row-label-mode", "value"),
    Input("cofit-column-label-mode", "value"),
)
def update_heatmap_section(
    cor_method,
    heatmap_gene_mode,
    n_top_genes,
    n_top_genes_n_submit,
    n_top_genes_n_blur,
    cluster_genes,
    heatmap_scale,
    row_label_mode,
    column_label_mode,
):
    try:
        fig, sub, metadata = make_heatmap_figure(
            cor_method=cor_method,
            heatmap_gene_mode=heatmap_gene_mode,
            n_top_genes=n_top_genes,
            cluster_genes=cluster_genes,
            scale_method=heatmap_scale,
            row_label_mode=row_label_mode,
            col_label_mode=column_label_mode,
        )

        requested_n_top = safe_int(
            n_top_genes,
            default=DEFAULT_HEATMAP_N_TOP,
            minimum=1,
            maximum=min(MAX_HEATMAP_N_GENES, metadata["total_genes"]),
        )

        status = dbc.Alert(
            [
                html.Strong("Current heatmap: "),
                html.Span(
                    f"Displaying {metadata['n_display']} of {metadata['total_genes']} genes "
                    f"({metadata['gene_selection_label']}). "
                ),
                html.Span(
                    f"Requested top-variable genes: {requested_n_top}. "
                    if heatmap_gene_mode == "top" else
                    f"Requested bottom-variable genes: {requested_n_top}. "
                ),
                html.Span(
                    "Values above 1,500 are capped at 1,500. Download the matrix table if you want to plot the full 3,330-gene matrix locally. "
                    if metadata["total_genes"] > MAX_HEATMAP_N_GENES else ""
                ),
                html.Span(metadata["label_note"]),
            ],
            color="light",
            className="mb-3",
        )

        if sub.shape[0] > MAX_DISPLAY_TABLE_DIM:
            table = html.Div(
                [
                    heatmap_table_annotation(metadata),
                    dbc.Alert(
                        (
                            f"The selected heatmap contains {sub.shape[0]} × {sub.shape[1]} values. "
                            "To keep the page responsive, the matrix table is not rendered in the browser. "
                            "Use the download button to export the full selected matrix."
                        ),
                        color="warning",
                        className="mb-3",
                    ),
                ]
            )
        else:
            df_display = sub.round(3).copy()
            df_display.columns = [display_gene_label(g) for g in df_display.columns]
            df_display.insert(0, "GeneName", [display_gene_name(g) for g in sub.index])
            df_display.insert(0, "GeneID", sub.index)

            table = html.Div(
                [
                    heatmap_table_annotation(metadata),

                    dash_table.DataTable(
                        data=df_display.to_dict("records"),
                        columns=[{"name": c, "id": c} for c in df_display.columns],
                        page_size=10,
                        filter_action="native",
                        sort_action="native",
                        style_table={
                            "overflowX": "auto",
                            "maxHeight": "520px",
                            "overflowY": "auto"
                        },
                        style_cell={
                            "textAlign": "left",
                            "fontFamily": "Arial",
                            "fontSize": "13px",
                            "padding": "5px",
                            "minWidth": "100px",
                            "whiteSpace": "normal",
                        },
                        style_header={
                            "fontWeight": "bold",
                            "backgroundColor": "#f1f3f5",
                        },
                    ),
                ]
            )

        return fig, table, status

    except Exception as e:
        fig = go.Figure()
        fig.update_layout(
            template="plotly_white",
            title="Error",
            annotations=[
                dict(
                    text=str(e),
                    x=0.5,
                    y=0.5,
                    showarrow=False
                )
            ]
        )
        return fig, dbc.Alert(str(e), color="danger"), ""


@dash.callback(
    Output("cofit-summary-stats", "children"),
    Input("cofit-cor-method", "value"),
)
def update_summary_stats(cor_method):
    try:
        summary_text = make_summary_stats(cor_method)

        return html.Div(
            [
                summary_annotation(),
                html.Pre(summary_text),
            ]
        )

    except Exception as e:
        return dbc.Alert(str(e), color="danger")


@dash.callback(
    Output("cofit-download-single-table", "data"),
    Input("cofit-download-single-table-button", "n_clicks"),
    State("cofit-cor-method", "value"),
    State("cofit-gene-select", "value"),
    State("cofit-top-n", "value"),
    State("cofit-correlation-direction", "value"),
    prevent_initial_call=True,
)
def download_single_gene_table(n_clicks, cor_method, gene_select, top_n, direction):
    if not n_clicks:
        return no_update

    df = get_single_gene_correlation_table(
        cor_method=cor_method,
        gene_id=gene_select,
        top_n=top_n,
        direction=direction,
    )

    safe_top_n = safe_int(top_n, default=20, minimum=1, maximum=MAX_SINGLE_GENE_TOP_N)
    filename = f"cofitness_{cor_method}_{gene_select}_top_{safe_top_n}_{direction}.csv"

    return dcc.send_data_frame(df.to_csv, filename, index=False)


@dash.callback(
    Output("cofit-download-heatmap-table", "data"),
    Input("cofit-download-heatmap-table-button", "n_clicks"),
    State("cofit-cor-method", "value"),
    State("cofit-heatmap-gene-mode", "value"),
    State("cofit-n-top-genes", "value"),
    State("cofit-cluster-genes", "value"),
    State("cofit-heatmap-scale", "value"),
    State("cofit-row-label-mode", "value"),
    State("cofit-column-label-mode", "value"),
    prevent_initial_call=True,
)
def download_heatmap_table(
    n_clicks,
    cor_method,
    heatmap_gene_mode,
    n_top_genes,
    cluster_genes,
    heatmap_scale,
    row_label_mode,
    column_label_mode,
):
    if not n_clicks:
        return no_update

    fig, sub, metadata = make_heatmap_figure(
        cor_method=cor_method,
        heatmap_gene_mode=heatmap_gene_mode,
        n_top_genes=n_top_genes,
        cluster_genes=cluster_genes,
        scale_method=heatmap_scale,
        row_label_mode=row_label_mode,
        col_label_mode=column_label_mode,
    )

    df_out = sub.round(6).copy()
    df_out.insert(0, "GeneID", sub.index)
    df_out.insert(1, "GeneName", [display_gene_name(g) for g in sub.index])

    safe_n_top = safe_int(
        n_top_genes,
        default=DEFAULT_HEATMAP_N_TOP,
        minimum=1,
        maximum=min(MAX_HEATMAP_N_GENES, metadata["total_genes"]),
    )
    selection_prefix = "bottom" if heatmap_gene_mode == "bottom" else "top"
    selection_label = f"{selection_prefix}_{safe_n_top}"

    filename = f"cofitness_heatmap_matrix_{cor_method}_{selection_label}_{cluster_genes}_{heatmap_scale}.csv"

    return dcc.send_data_frame(df_out.to_csv, filename, index=False)
