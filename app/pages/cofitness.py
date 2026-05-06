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
    "cofitness_correlation_matrix_spearman.csv"
)

PEARSON_FILE = os.path.join(
    COFIT_DIR,
    "cofitness_correlation_matrix_pearson.csv"
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
    return mats[cor_method]


def get_cor_label(cor_method):
    return "Spearman" if cor_method == "spearman" else "Pearson"


# ============================================================
# 2. Heatmap helpers
# ============================================================

def get_top_variable_genes(mat, n_top):
    gene_sd = mat.std(axis=1, skipna=True)
    n = min(int(n_top), len(gene_sd))
    return gene_sd.sort_values(ascending=False).head(n).index.tolist()


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


def make_heatmap_figure(cor_method, n_top_genes, cluster_genes, scale_method):
    mat = get_active_matrix(cor_method)
    cor_label = get_cor_label(cor_method)

    top_genes = get_top_variable_genes(mat, n_top_genes)
    sub = mat.loc[top_genes, top_genes].copy()

    raw_min = float(np.nanmin(sub.values))
    raw_max = float(np.nanmax(sub.values))

    if cluster_genes == "yes":
        sub = cluster_matrix(sub)

    zmin, zmax, scale_label = get_heatmap_limits(sub, scale_method)

    x_labels = [display_gene_label(g) for g in sub.columns]
    y_labels = [display_gene_label(g) for g in sub.index]

    fig = go.Figure(
        data=go.Heatmap(
            x=x_labels,
            y=y_labels,
            z=sub.values,
            colorscale=[
                [0.0, "blue"],
                [0.5, "white"],
                [1.0, "red"],
            ],
            zmin=zmin,
            zmax=zmax,
            zauto=False,
            colorbar=dict(
                title=f"{cor_label}<br>correlation",
                tickformat=".2f"
            ),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "vs<br>"
                "<b>%{x}</b><br>"
                f"{cor_label} correlation: "
                "%{z:.3f}<extra></extra>"
            )
        )
    )

    fig.update_layout(
        title=(
            f"{cor_label} Co-fitness Correlation Matrix<br>"
            f"<sup>Top {sub.shape[0]} variable genes | "
            f"Raw range: [{raw_min:.3f}, {raw_max:.3f}] | "
            f"{scale_label}</sup>"
        ),
        template="plotly_white",
        height=760,
        margin=dict(l=200, r=60, t=90, b=200),
        xaxis=dict(tickangle=-45, tickfont=dict(size=8)),
        yaxis=dict(tickfont=dict(size=8)),
    )

    return fig, sub


# ============================================================
# 3. Single-gene cofitness helpers
# ============================================================

def get_single_gene_correlation_table(cor_method, gene_id, top_n, direction):
    mat = get_active_matrix(cor_method)
    cor_label = get_cor_label(cor_method)

    if gene_id not in mat.index:
        raise ValueError(f"Gene not found in correlation matrix: {gene_id}")

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

    v = v.head(int(top_n))

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

    gene_sd = mat.std(axis=1, skipna=True).sort_values(ascending=False)

    top_var = pd.DataFrame({
        "GeneID": gene_sd.head(10).index,
        "GeneName": [display_gene_name(g) for g in gene_sd.head(10).index],
        "SD_across_correlations": gene_sd.head(10).values,
    })

    vals = mat.values.flatten()
    vals = vals[np.isfinite(vals)]

    summary_text = (
        f"{cor_label} Co-fitness Correlation Matrix Summary\n"
        f"======================================\n\n"
        f"Matrix dimensions: {mat.shape[0]} x {mat.shape[1]}\n"
        f"Number of genes: {mat.shape[0]}\n\n"
        f"Correlation range: {np.nanmin(vals):.3f} to {np.nanmax(vals):.3f}\n"
        f"Mean correlation: {np.nanmean(vals):.3f}\n"
        f"Median correlation: {np.nanmedian(vals):.3f}\n\n"
        f"Top 10 most variable genes under {cor_label} correlation:"
    )

    return summary_text, top_var


# ============================================================
# 5. Annotation text blocks
# ============================================================

def heatmap_table_annotation():
    return dbc.Alert(
        [
            html.Strong("About this co-fitness matrix table: "),
            html.Span(
                "Each row and column represents a gene. Each cell shows the selected "
                "co-fitness correlation value between the row gene and the column gene. "
                "Positive values indicate similar fitness behavior across conditions, while "
                "negative values indicate opposite fitness behavior. Values close to 0 indicate "
                "weak or no co-fitness relationship."
            ),
            html.Br(),
            html.Span(
                "This table corresponds to the genes currently displayed in the heatmap. "
                "If clustering is enabled, the row and column order follows the hierarchical "
                "clustering used in the heatmap."
            ),
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
# 6. Layout
# ============================================================

try:
    _, _, _, gene_dropdown_options = load_gene_lookup()
    default_gene = gene_dropdown_options[0]["value"] if gene_dropdown_options else None
except Exception:
    gene_dropdown_options = []
    default_gene = None


layout = dbc.Container(
    [
        html.H2("Cofitness", className="page-title"),

        html.P(
            "Explore co-fitness correlations between genes using Spearman or Pearson correlation matrices. "
            "The single-gene co-fitness plot is shown first, followed by the interactive co-fitness heatmap.",
            className="lead"
        ),

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
                    "or absolute co-fitness correlations. Positive correlations indicate similar "
                    "fitness profiles, while negative correlations indicate opposite fitness profiles."
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
                        dbc.Input(
                            id="cofit-top-n",
                            type="number",
                            min=5,
                            max=100,
                            step=5,
                            value=20,
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

        dbc.Alert(
            [
                html.Strong("How to use this module: "),
                html.Span(
                    "The heatmap shows pairwise co-fitness correlations among the top variable genes. "
                    "You can change the number of genes, enable hierarchical clustering, and adjust color scaling. "
                    "Zoom by selecting an area; hover over cells to see exact correlation values; double-click to reset."
                ),
            ],
            color="secondary",
            className="mb-3",
        ),

        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Label("Number of top variable genes"),
                        dbc.Input(
                            id="cofit-n-top-genes",
                            type="number",
                            min=5,
                            max=100,
                            step=5,
                            value=20,
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
                        ),
                    ],
                    md=3,
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
                    md=4,
                ),
            ],
            className="mb-3",
        ),

        dbc.Row(
            [
                dbc.Col(
                    [
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
# 7. Callbacks
# ============================================================

@dash.callback(
    Output("cofit-single-gene-plot", "figure"),
    Output("cofit-single-gene-table", "children"),
    Input("cofit-cor-method", "value"),
    Input("cofit-gene-select", "value"),
    Input("cofit-top-n", "value"),
    Input("cofit-correlation-direction", "value"),
)
def update_single_gene_section(
    cor_method,
    gene_select,
    top_n,
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
    Input("cofit-cor-method", "value"),
    Input("cofit-n-top-genes", "value"),
    Input("cofit-cluster-genes", "value"),
    Input("cofit-heatmap-scale", "value"),
)
def update_heatmap_section(
    cor_method,
    n_top_genes,
    cluster_genes,
    heatmap_scale,
):
    try:
        fig, sub = make_heatmap_figure(
            cor_method=cor_method,
            n_top_genes=n_top_genes,
            cluster_genes=cluster_genes,
            scale_method=heatmap_scale,
        )

        df_display = sub.round(3).copy()
        df_display.columns = [display_gene_label(g) for g in df_display.columns]
        df_display.insert(0, "GeneName", [display_gene_name(g) for g in sub.index])
        df_display.insert(0, "GeneID", sub.index)

        table = html.Div(
            [
                heatmap_table_annotation(),

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
    Output("cofit-summary-stats", "children"),
    Input("cofit-cor-method", "value"),
)
def update_summary_stats(cor_method):
    try:
        summary_text, top_var = make_summary_stats(cor_method)

        return html.Div(
            [
                summary_annotation(),

                html.Pre(summary_text),

                dash_table.DataTable(
                    data=top_var.round(4).to_dict("records"),
                    columns=[{"name": c, "id": c} for c in top_var.columns],
                    page_size=10,
                    style_table={"overflowX": "auto"},
                    style_cell={
                        "textAlign": "left",
                        "fontFamily": "Arial",
                        "fontSize": "14px",
                        "padding": "6px",
                    },
                    style_header={
                        "fontWeight": "bold",
                        "backgroundColor": "#f1f3f5",
                    },
                ),
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

    filename = f"cofitness_{cor_method}_{gene_select}_top_{top_n}_{direction}.csv"

    return dcc.send_data_frame(df.to_csv, filename, index=False)


@dash.callback(
    Output("cofit-download-heatmap-table", "data"),
    Input("cofit-download-heatmap-table-button", "n_clicks"),
    State("cofit-cor-method", "value"),
    State("cofit-n-top-genes", "value"),
    State("cofit-cluster-genes", "value"),
    State("cofit-heatmap-scale", "value"),
    prevent_initial_call=True,
)
def download_heatmap_table(n_clicks, cor_method, n_top_genes, cluster_genes, heatmap_scale):
    if not n_clicks:
        return no_update

    fig, sub = make_heatmap_figure(
        cor_method=cor_method,
        n_top_genes=n_top_genes,
        cluster_genes=cluster_genes,
        scale_method=heatmap_scale,
    )

    df_out = sub.round(6).copy()
    df_out.insert(0, "GeneID", sub.index)
    df_out.insert(1, "GeneName", [display_gene_name(g) for g in sub.index])

    filename = f"cofitness_heatmap_matrix_{cor_method}_top_{n_top_genes}.csv"

    return dcc.send_data_frame(df_out.to_csv, filename, index=False)
