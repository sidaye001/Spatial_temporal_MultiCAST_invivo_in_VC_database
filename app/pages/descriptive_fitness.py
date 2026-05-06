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


dash.register_page(__name__, path="/descriptive-fitness", name="Descriptive fitness")


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

GENE_SET_DIR = os.path.join(DATA_DIR, "gene_sets")

DEFAULT_SINGLE_GENE_QUERY = "motV"
DEFAULT_GENE_SET_FILE = "Tcp_genes.csv"

TIME_LEVELS = ["1h", "3h", "6h", "12h", "24h"]

SPACE_LEVELS = [
    "st", "SI1", "SI2", "SI3", "SI4", "SI5",
    "SI6", "SI7", "SI8", "SI9", "ce", "co"
]

TIME_NUM_MAP = {
    "1h": 1,
    "3h": 3,
    "6h": 6,
    "12h": 12,
    "24h": 24,
}

SPACE_NUM_MAP = {s: i + 1 for i, s in enumerate(SPACE_LEVELS)}

CONDITION_ORDER = [
    f"{t}_{s}"
    for t in TIME_LEVELS
    for s in SPACE_LEVELS
]

# Brown/blue diverging scale with narrow white center.
# This makes the color transition faster and avoids a wide white neutral range.
NARROW_WHITE_BROWN_BLUE = [
    [0.00, "#08306B"],
    [0.18, "#2171B5"],
    [0.36, "#6BAED6"],
    [0.47, "#DDEEFF"],
    [0.498, "#FFFFFF"],
    [0.502, "#FFFFFF"],
    [0.53, "#FEE8C8"],
    [0.64, "#FDBB84"],
    [0.82, "#E34A33"],
    [1.00, "#7F0000"],
]


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

    if "Beta" not in df.columns:
        if "logFC" in df.columns:
            df = df.rename(columns={"logFC": "Beta"})
        else:
            raise ValueError("Raw data must contain either 'Beta' or 'logFC' column.")

    df = df[["Gene", "Time", "Space", "Beta"]].copy()

    df["Gene"] = df["Gene"].astype(str).str.strip()
    df["Time"] = df["Time"].astype(str).str.strip()
    df["Space"] = df["Space"].astype(str).str.strip()
    df["Beta"] = pd.to_numeric(df["Beta"], errors="coerce")

    df = df[df["Time"].isin(TIME_LEVELS)]
    df = df[df["Space"].isin(SPACE_LEVELS)]
    df = df.dropna(subset=["Gene", "Time", "Space", "Beta"]).copy()

    df["Time"] = pd.Categorical(df["Time"], categories=TIME_LEVELS, ordered=True)
    df["Space"] = pd.Categorical(df["Space"], categories=SPACE_LEVELS, ordered=True)

    df["TimeNum"] = df["Time"].astype(str).map(TIME_NUM_MAP)
    df["SpaceNum"] = df["Space"].astype(str).map(SPACE_NUM_MAP)
    df["Condition"] = df["Time"].astype(str) + "_" + df["Space"].astype(str)

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

    ann["GeneName"] = np.where(
        ann["gene_name"] != "",
        ann["gene_name"],
        ann["locus_ID"]
    )

    ann = (
        ann.rename(columns={"locus_ID": "Gene"})
        [["Gene", "GeneName"]]
        .drop_duplicates()
    )

    return ann


@lru_cache(maxsize=1)
def load_gene_lookup():
    raw = load_raw_data()
    ann = load_annotation()

    lookup = pd.DataFrame({"Gene": sorted(raw["Gene"].unique())})
    lookup = lookup.merge(ann, on="Gene", how="left")
    lookup["GeneName"] = lookup["GeneName"].fillna(lookup["Gene"])
    lookup["DisplayLabel"] = lookup["GeneName"] + " | " + lookup["Gene"]

    gene_to_name = dict(zip(lookup["Gene"], lookup["GeneName"]))

    gene_upper_to_gene = {
        g.upper(): g
        for g in lookup["Gene"]
    }

    name_lower_to_genes = {}
    for _, row in lookup.iterrows():
        nm = str(row["GeneName"]).strip().lower()
        if nm:
            name_lower_to_genes.setdefault(nm, []).append(row["Gene"])

    options = [
        {"label": row["DisplayLabel"], "value": row["Gene"]}
        for _, row in lookup.sort_values(["GeneName", "Gene"]).iterrows()
    ]

    return lookup, gene_to_name, gene_upper_to_gene, name_lower_to_genes, options


def add_annotation(df):
    ann = load_annotation()
    out = df.copy()

    annotation_like_cols = [
        "GeneName", "gene_name", "GeneName_x", "GeneName_y",
        "gene_name_x", "gene_name_y"
    ]

    drop_cols = [c for c in annotation_like_cols if c in out.columns]
    if drop_cols:
        out = out.drop(columns=drop_cols)

    out["Gene"] = out["Gene"].astype(str)
    out = out.merge(ann, on="Gene", how="left")
    out["GeneName"] = out["GeneName"].fillna(out["Gene"])

    return out


def get_gene_name(gene_id):
    _, gene_to_name, _, _, _ = load_gene_lookup()
    return gene_to_name.get(gene_id, gene_id)


def resolve_gene_token(token):
    """
    Resolve one user token as either Gene ID or GeneName.
    Returns a list of matching Gene IDs.
    """
    if token is None:
        return []

    token = str(token).strip()
    if token == "":
        return []

    _, _, gene_upper_to_gene, name_lower_to_genes, _ = load_gene_lookup()

    if token.upper() in gene_upper_to_gene:
        return [gene_upper_to_gene[token.upper()]]

    token_lower = token.lower()
    if token_lower in name_lower_to_genes:
        return name_lower_to_genes[token_lower]

    return []


def parse_gene_text(text):
    if text is None:
        return []

    text = str(text)

    for sep in [",", ";", "\n", "\t", "\r"]:
        text = text.replace(sep, " ")

    tokens = [x.strip() for x in text.split(" ") if x.strip() != ""]
    return tokens


def resolve_gene_list(tokens):
    found = []
    missing = []

    for token in tokens:
        matched = resolve_gene_token(token)
        if matched:
            found.extend(matched)
        else:
            missing.append(token)

    found = list(dict.fromkeys(found))

    return found, missing


# ============================================================
# 2. Gene set loading
# ============================================================

def list_gene_set_files():
    """
    Automatically include all CSV/TXT/TSV files in:
      ../data/gene_sets
    """
    if not os.path.exists(GENE_SET_DIR):
        return []

    files = [
        f for f in os.listdir(GENE_SET_DIR)
        if f.lower().endswith((".csv", ".txt", ".tsv"))
    ]

    return sorted(files, key=lambda x: x.lower())


def read_gene_set_file(filename):
    if filename is None or filename == "":
        return []

    path = os.path.join(GENE_SET_DIR, filename)

    if not os.path.exists(path):
        return []

    if filename.lower().endswith(".csv"):
        df = pd.read_csv(path)
    elif filename.lower().endswith(".tsv"):
        df = pd.read_csv(path, sep="\t")
    else:
        df = pd.read_csv(path, sep=None, engine="python", header=None)

    if df.shape[1] == 0:
        return []

    preferred_cols = [
        "Gene", "gene", "GeneID", "geneID", "locus_ID",
        "locus_id", "locus", "GeneName", "gene_name",
        "X", "x"
    ]

    selected_col = None
    for col in preferred_cols:
        if col in df.columns:
            selected_col = col
            break

    if selected_col is None:
        selected_col = df.columns[0]

    vals = (
        df[selected_col]
        .dropna()
        .astype(str)
        .str.strip()
        .tolist()
    )

    vals = [x for x in vals if x != ""]
    return vals


def parse_uploaded_gene_file(contents, filename):
    if contents is None:
        return []

    content_type, content_string = contents.split(",", 1)
    decoded = base64.b64decode(content_string)

    if filename and filename.lower().endswith(".csv"):
        df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
    elif filename and filename.lower().endswith(".tsv"):
        df = pd.read_csv(io.StringIO(decoded.decode("utf-8")), sep="\t")
    else:
        text = decoded.decode("utf-8")
        return parse_gene_text(text)

    if df.shape[1] == 0:
        return []

    preferred_cols = [
        "Gene", "gene", "GeneID", "geneID", "locus_ID",
        "locus_id", "GeneName", "gene_name", "X", "x"
    ]

    selected_col = None
    for col in preferred_cols:
        if col in df.columns:
            selected_col = col
            break

    if selected_col is None:
        selected_col = df.columns[0]

    return (
        df[selected_col]
        .dropna()
        .astype(str)
        .str.strip()
        .tolist()
    )


def get_gene_set_inputs(gene_set_file, manual_text, upload_contents, upload_filename):
    tokens = []

    predefined_tokens = read_gene_set_file(gene_set_file)
    manual_tokens = parse_gene_text(manual_text)

    upload_tokens = []
    if upload_contents is not None:
        upload_tokens = parse_uploaded_gene_file(upload_contents, upload_filename)

    tokens.extend(predefined_tokens)
    tokens.extend(manual_tokens)
    tokens.extend(upload_tokens)

    tokens = [str(x).strip() for x in tokens if str(x).strip() != ""]

    gene_ids, missing = resolve_gene_list(tokens)

    if gene_set_file:
        gene_set_label = os.path.splitext(os.path.basename(gene_set_file))[0]
    elif upload_filename:
        gene_set_label = os.path.splitext(os.path.basename(upload_filename))[0]
    else:
        gene_set_label = "Custom gene set"

    return gene_ids, missing, gene_set_label


# ============================================================
# 3. Single gene spatial-temporal plot
# ============================================================

def make_single_gene_surface_contour(gene_id):
    raw = load_raw_data()
    sub = raw[raw["Gene"] == gene_id].copy()

    if sub.empty:
        raise ValueError(f"Gene not found in raw data: {gene_id}")

    sub["Time"] = pd.Categorical(sub["Time"].astype(str), categories=TIME_LEVELS, ordered=True)
    sub["Space"] = pd.Categorical(sub["Space"].astype(str), categories=SPACE_LEVELS, ordered=True)

    pivot = (
        sub
        .pivot_table(
            index="Space",
            columns="Time",
            values="Beta",
            aggfunc="mean",
            observed=False
        )
        .reindex(index=SPACE_LEVELS, columns=TIME_LEVELS)
    )

    z = pivot.values.astype(float)

    if np.isnan(z).all():
        raise ValueError(f"No finite Beta values found for gene: {gene_id}")

    z_plot = pd.DataFrame(z).interpolate(axis=0).interpolate(axis=1).fillna(0).values

    min_idx = np.unravel_index(np.nanargmin(z_plot), z_plot.shape)
    max_idx = np.unravel_index(np.nanargmax(z_plot), z_plot.shape)

    min_space = SPACE_LEVELS[min_idx[0]]
    min_time = TIME_LEVELS[min_idx[1]]
    min_beta = z_plot[min_idx]

    max_space = SPACE_LEVELS[max_idx[0]]
    max_time = TIME_LEVELS[max_idx[1]]
    max_beta = z_plot[max_idx]

    gene_name = get_gene_name(gene_id)
    display_label = f"{gene_name} | {gene_id}"

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "surface"}, {"type": "xy"}]],
        subplot_titles=[
            f"3D Fitness Surface: {display_label}",
            f"Contour Map: {display_label}"
        ],
        horizontal_spacing=0.08,
    )

    fig.add_trace(
        go.Surface(
            x=[TIME_NUM_MAP[t] for t in TIME_LEVELS],
            y=list(range(1, len(SPACE_LEVELS) + 1)),
            z=z_plot,
            colorscale=NARROW_WHITE_BROWN_BLUE,
            colorbar=dict(
                title="Beta",
                x=0.43,
                len=0.7,
            ),
            contours={
                "z": {
                    "show": True,
                    "usecolormap": True,
                    "highlightcolor": "white",
                    "project_z": True,
                }
            },
            hovertemplate=(
                "Time index: %{x}<br>"
                "Space index: %{y}<br>"
                "Beta: %{z:.3f}<extra></extra>"
            ),
            showscale=True,
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Heatmap(
            x=TIME_LEVELS,
            y=SPACE_LEVELS,
            z=z_plot,
            colorscale=NARROW_WHITE_BROWN_BLUE,
            colorbar=dict(title="Beta", x=1.02),
            hovertemplate=(
                "Time: %{x}<br>"
                "Space: %{y}<br>"
                "Beta: %{z:.3f}<extra></extra>"
            ),
        ),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Contour(
            x=TIME_LEVELS,
            y=SPACE_LEVELS,
            z=z_plot,
            contours=dict(
                coloring="none",
                showlabels=False,
            ),
            line=dict(color="black", width=1.5),
            showscale=False,
            hoverinfo="skip",
        ),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Scatter(
            x=[max_time],
            y=[max_space],
            mode="markers+text",
            marker=dict(size=12, color="yellow", line=dict(color="black", width=1)),
            text=[f"Max: {max_beta:.2f}"],
            textposition="top center",
            textfont=dict(color="yellow", size=14),
            name="Maximum Beta",
            hovertemplate=(
                f"Maximum Beta<br>Time: {max_time}<br>"
                f"Space: {max_space}<br>Beta: {max_beta:.3f}<extra></extra>"
            ),
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Scatter(
            x=[min_time],
            y=[min_space],
            mode="markers+text",
            marker=dict(size=12, color="lime", line=dict(color="black", width=1)),
            text=[f"Min: {min_beta:.2f}"],
            textposition="bottom center",
            textfont=dict(color="lime", size=14),
            name="Minimum Beta",
            hovertemplate=(
                f"Minimum Beta<br>Time: {min_time}<br>"
                f"Space: {min_space}<br>Beta: {min_beta:.3f}<extra></extra>"
            ),
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    fig.update_scenes(
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
        zaxis=dict(title="Beta"),
        camera=dict(eye=dict(x=1.8, y=1.5, z=1.1)),
        row=1,
        col=1,
    )

    fig.update_xaxes(title_text="Time", row=1, col=2)
    fig.update_yaxes(
        title_text="Space",
        categoryorder="array",
        categoryarray=SPACE_LEVELS,
        row=1,
        col=2,
    )

    fig.update_layout(
        template="plotly_white",
        height=720,
        margin=dict(l=30, r=40, t=90, b=40),
        title=f"Spatial-temporal in vivo fitness landscape for {display_label}",
    )

    summary = dbc.Alert(
        [
            html.Strong("Single-gene fitness landscape summary"),
            html.Br(),
            f"Query gene: {display_label}",
            html.Br(),
            f"Maximum Beta: {max_beta:.3f} at {max_time}, {max_space}",
            html.Br(),
            f"Minimum Beta: {min_beta:.3f} at {min_time}, {min_space}",
            html.Br(),
            html.Span(
                "The 3D surface and 2D contour map show how the fitness coefficient changes "
                "across infection time and intestinal location."
            ),
        ],
        color="info",
        className="mb-3",
    )

    return fig, summary


# ============================================================
# 4. Gene set descriptive plots
# ============================================================

def get_gene_set_matrix(gene_ids):
    raw = load_raw_data()
    sub = raw[raw["Gene"].isin(gene_ids)].copy()

    if sub.empty:
        raise ValueError("None of the selected genes were found in the raw fitness data.")

    sub = add_annotation(sub)

    sub["Time"] = pd.Categorical(sub["Time"].astype(str), categories=TIME_LEVELS, ordered=True)
    sub["Space"] = pd.Categorical(sub["Space"].astype(str), categories=SPACE_LEVELS, ordered=True)
    sub["Condition"] = sub["Time"].astype(str) + "_" + sub["Space"].astype(str)

    gene_order = (
        sub[["Gene", "GeneName"]]
        .drop_duplicates()
        .sort_values(["GeneName", "Gene"])
    )

    ordered_genes = gene_order["Gene"].astype(str).tolist()

    mat = (
        sub
        .pivot_table(
            index="Gene",
            columns="Condition",
            values="Beta",
            aggfunc="mean",
            observed=False
        )
        .reindex(index=ordered_genes, columns=CONDITION_ORDER)
    )

    label_map = dict(zip(gene_order["Gene"], gene_order["GeneName"]))
    y_labels = [label_map.get(g, g) for g in mat.index]

    return sub, mat, y_labels


def make_gene_set_descriptive_figure(gene_ids, gene_set_label):
    sub, mat, y_labels = get_gene_set_matrix(gene_ids)

    n_genes = sub["Gene"].nunique()

    temporal = (
        sub
        .groupby("Time", observed=False)["Beta"]
        .agg(["mean", "sem"])
        .reindex(TIME_LEVELS)
        .reset_index()
    )
    temporal["sem"] = temporal["sem"].fillna(0)

    spatial = (
        sub
        .groupby("Space", observed=False)["Beta"]
        .agg(["mean", "sem"])
        .reindex(SPACE_LEVELS)
        .reset_index()
    )
    spatial["sem"] = spatial["sem"].fillna(0)

    interaction = (
        sub
        .pivot_table(
            index="Time",
            columns="Space",
            values="Beta",
            aggfunc="mean",
            observed=False
        )
        .reindex(index=TIME_LEVELS, columns=SPACE_LEVELS)
    )

    ranked = (
        sub
        .groupby(["Gene", "GeneName"], observed=False)["Beta"]
        .mean()
        .reset_index()
        .sort_values("Beta")
    )
    ranked["Label"] = ranked["GeneName"].fillna(ranked["Gene"])

    fig = make_subplots(
        rows=4,
        cols=3,
        specs=[
            [{"type": "heatmap", "colspan": 3}, None, None],
            [{"type": "xy"}, {"type": "xy"}, {"type": "heatmap"}],
            [{"type": "xy", "colspan": 3}, None, None],
            [{"type": "xy", "colspan": 3}, None, None],
        ],
        subplot_titles=[
            f"{gene_set_label}: Gene fitness across all conditions",
            "Temporal pattern",
            "Spatial pattern",
            "Space × time interaction",
            f"{gene_set_label}: genes ranked by mean fitness",
            f"{gene_set_label}: beta distribution by intestinal location",
        ],
        vertical_spacing=0.11,
        horizontal_spacing=0.08,
    )

    z = mat.values.astype(float)
    finite_z = z[np.isfinite(z)]

    if len(finite_z) > 0:
        abs_lim = np.nanpercentile(np.abs(finite_z), 95)
        if abs_lim == 0:
            abs_lim = np.nanmax(np.abs(finite_z))
        if abs_lim == 0:
            abs_lim = 1
        zmin, zmax = -abs_lim, abs_lim
    else:
        zmin, zmax = -1, 1

    fig.add_trace(
        go.Heatmap(
            z=z,
            x=CONDITION_ORDER,
            y=y_labels,
            colorscale=NARROW_WHITE_BROWN_BLUE,
            zmin=zmin,
            zmax=zmax,
            zmid=0,
            colorbar=dict(title="Beta", x=1.02, y=0.86, len=0.28),
            hovertemplate=(
                "Gene: %{y}<br>"
                "Condition: %{x}<br>"
                "Beta: %{z:.3f}<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=temporal["Time"].astype(str),
            y=temporal["mean"],
            mode="lines+markers",
            line=dict(width=3),
            marker=dict(size=8),
            error_y=dict(
                type="data",
                array=temporal["sem"],
                visible=True,
            ),
            name="Temporal mean Beta",
            hovertemplate="Time: %{x}<br>Mean Beta: %{y:.3f}<extra></extra>",
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="gray",
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Bar(
            x=spatial["Space"].astype(str),
            y=spatial["mean"],
            error_y=dict(
                type="data",
                array=spatial["sem"],
                visible=True,
            ),
            marker_line=dict(width=0.5, color="black"),
            name="Spatial mean Beta",
            hovertemplate="Space: %{x}<br>Mean Beta: %{y:.3f}<extra></extra>",
            showlegend=False,
        ),
        row=2,
        col=2,
    )

    fig.add_trace(
        go.Heatmap(
            z=interaction.values,
            x=SPACE_LEVELS,
            y=TIME_LEVELS,
            colorscale=NARROW_WHITE_BROWN_BLUE,
            zmid=0,
            colorbar=dict(title="Mean Beta", x=1.02, y=0.48, len=0.25),
            hovertemplate=(
                "Space: %{x}<br>"
                "Time: %{y}<br>"
                "Mean Beta: %{z:.3f}<extra></extra>"
            ),
        ),
        row=2,
        col=3,
    )

    bar_colors = np.where(ranked["Beta"] >= 0, "#25BBD3", "#D94B55")
    fig.add_trace(
        go.Bar(
            x=ranked["Beta"],
            y=ranked["Label"],
            orientation="h",
            marker=dict(color=bar_colors, line=dict(color="black", width=0.5)),
            text=np.round(ranked["Beta"], 2),
            textposition="outside",
            hovertemplate=(
                "Gene: %{y}<br>"
                "Mean Beta: %{x:.3f}<extra></extra>"
            ),
            showlegend=False,
        ),
        row=3,
        col=1,
    )

    fig.add_vline(
        x=0,
        line_color="black",
        line_width=1,
        row=3,
        col=1,
    )

    for s in SPACE_LEVELS:
        vals = sub[sub["Space"].astype(str) == s]["Beta"].dropna()
        fig.add_trace(
            go.Box(
                y=vals,
                name=s,
                boxpoints="outliers",
                marker=dict(size=3),
                line=dict(width=1),
                showlegend=False,
                hovertemplate=f"Space: {s}<br>Beta: %{{y:.3f}}<extra></extra>",
            ),
            row=4,
            col=1,
        )

    fig.update_xaxes(
        tickangle=90,
        title_text="Condition (timepoint_space)",
        row=1,
        col=1,
    )
    fig.update_yaxes(title_text="Gene", row=1, col=1)

    fig.update_xaxes(title_text="Timepoint", row=2, col=1)
    fig.update_yaxes(title_text="Mean Beta coefficient", row=2, col=1)

    fig.update_xaxes(title_text="Intestinal location", tickangle=45, row=2, col=2)
    fig.update_yaxes(title_text="Mean Beta coefficient", row=2, col=2)

    fig.update_xaxes(title_text="Location", tickangle=45, row=2, col=3)
    fig.update_yaxes(title_text="Timepoint", row=2, col=3)

    fig.update_xaxes(title_text="Mean Beta coefficient", row=3, col=1)
    fig.update_yaxes(title_text="Gene", row=3, col=1)

    fig.update_xaxes(title_text="Intestinal location", row=4, col=1)
    fig.update_yaxes(title_text="Beta coefficient", row=4, col=1)

    fig.update_layout(
        title=f"Descriptive fitness summary for {gene_set_label} ({n_genes} genes found)",
        template="plotly_white",
        height=1500,
        margin=dict(l=90, r=90, t=110, b=80),
    )

    summary = dbc.Alert(
        [
            html.Strong("Gene-set summary"),
            html.Br(),
            f"Input gene set: {gene_set_label}",
            html.Br(),
            f"Number of genes found in dataset: {n_genes}",
            html.Br(),
            html.Span(
                "The top heatmap shows each gene across all spatial-temporal conditions. "
                "The temporal, spatial, and interaction panels summarize the average pattern "
                "across the selected gene set. The ranked bar plot orders genes by mean Beta "
                "coefficient, and the box plot shows the distribution of Beta values across intestinal locations."
            ),
        ],
        color="info",
        className="mb-3",
    )

    table = (
        sub[["Gene", "GeneName", "Time", "Space", "Beta", "Condition"]]
        .sort_values(["GeneName", "Gene", "Time", "Space"])
        .copy()
    )

    return fig, summary, table


# ============================================================
# 5. Layout defaults
# ============================================================

try:
    _, _, _, _, gene_options = load_gene_lookup()
    default_gene_matches = resolve_gene_token(DEFAULT_SINGLE_GENE_QUERY)
    default_single_gene = default_gene_matches[0] if default_gene_matches else (
        gene_options[0]["value"] if gene_options else None
    )
except Exception:
    gene_options = []
    default_single_gene = None

gene_set_files = list_gene_set_files()

default_gene_set_file = DEFAULT_GENE_SET_FILE if DEFAULT_GENE_SET_FILE in gene_set_files else (
    gene_set_files[0] if gene_set_files else None
)


# ============================================================
# 6. Layout
# ============================================================

layout = dbc.Container(
    [
        html.H2("Descriptive fitness", className="page-title"),

        html.P(
            "Explore spatial-temporal in vivo fitness patterns for individual genes and gene sets. "
            "This page is divided into two modules: single-gene fitness landscape visualization and "
            "gene-set descriptive fitness summaries.",
            className="lead",
        ),

        dbc.Alert(
            [
                html.Strong("How to use this page: "),
                html.Span(
                    "Use the first module to inspect one gene across infection time and intestinal location. "
                    "Use the second module to summarize a predefined or uploaded gene set across all spatial-temporal "
                    "conditions. Beta represents the fitness coefficient used in the dataset."
                ),
            ],
            color="secondary",
            className="mb-4",
        ),

        html.H4("1. Single-gene spatial-temporal fitness landscape"),

        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Label("Search single gene by Gene ID or GeneName"),
                        dcc.Dropdown(
                            id="desc-single-gene",
                            options=gene_options,
                            value=default_single_gene,
                            searchable=True,
                            clearable=False,
                        ),
                    ],
                    md=6,
                ),
            ],
            className="mb-3",
        ),

        dcc.Loading(
            type="circle",
            children=[
                html.Div(id="desc-single-gene-summary"),
                dcc.Graph(id="desc-single-gene-plot", style={"width": "100%"}),
            ],
        ),

        html.Hr(),

        html.H4("2. Gene-set descriptive fitness summary"),

        dbc.Alert(
            [
                html.Strong("Input options: "),
                html.Span(
                    "Choose any predefined gene set from ../data/gene_sets, paste gene IDs or gene names, "
                    "or upload a CSV/TXT/TSV file. Manual input and uploaded genes are added to the predefined set."
                ),
            ],
            color="secondary",
            className="mb-3",
        ),

        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Label("Predefined gene set"),
                        dcc.Dropdown(
                            id="desc-gene-set-file",
                            options=[
                                {"label": f, "value": f}
                                for f in gene_set_files
                            ],
                            value=default_gene_set_file,
                            clearable=True,
                            placeholder="Choose a predefined gene set",
                        ),
                    ],
                    md=4,
                ),
                dbc.Col(
                    [
                        html.Label("Manual gene input"),
                        dcc.Textarea(
                            id="desc-gene-set-text",
                            placeholder="Type gene IDs or gene names separated by comma, space, or new line",
                            value="",
                            style={
                                "width": "100%",
                                "height": "90px",
                            },
                        ),
                    ],
                    md=4,
                ),
                dbc.Col(
                    [
                        html.Label("Upload gene list"),
                        dcc.Upload(
                            id="desc-gene-set-upload",
                            children=html.Div(
                                [
                                    "Drag and drop or ",
                                    html.A("select a CSV/TXT/TSV file"),
                                ]
                            ),
                            style={
                                "width": "100%",
                                "height": "90px",
                                "lineHeight": "90px",
                                "borderWidth": "1px",
                                "borderStyle": "dashed",
                                "borderRadius": "8px",
                                "textAlign": "center",
                                "backgroundColor": "#f8f9fa",
                            },
                            multiple=False,
                        ),
                        html.Div(id="desc-upload-filename", className="text-muted mt-2"),
                    ],
                    md=4,
                ),
            ],
            className="mb-3",
        ),

        dbc.Row(
            [
                dbc.Col(
                    dbc.Button(
                        "Update gene-set descriptive plots",
                        id="desc-update-gene-set-button",
                        color="primary",
                        className="mb-3",
                    ),
                    md="auto",
                ),
            ],
            className="mb-3",
        ),

        dcc.Loading(
            type="circle",
            children=[
                html.Div(id="desc-gene-set-summary"),
                dcc.Graph(id="desc-gene-set-plot", style={"width": "100%"}),
                html.Div(id="desc-gene-set-table-wrapper"),
            ],
        ),

        dcc.Download(id="desc-download-gene-set-table"),
    ],
    fluid=True,
)


# ============================================================
# 7. Callbacks
# ============================================================

@dash.callback(
    Output("desc-single-gene-plot", "figure"),
    Output("desc-single-gene-summary", "children"),
    Input("desc-single-gene", "value"),
)
def update_single_gene_plot(gene_id):
    try:
        if gene_id is None:
            fig = go.Figure()
            fig.update_layout(template="plotly_white", title="No gene selected")
            return fig, dbc.Alert("No gene selected.", color="warning")

        fig, summary = make_single_gene_surface_contour(gene_id)
        return fig, summary

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
                    showarrow=False,
                )
            ],
        )
        return fig, dbc.Alert(str(e), color="danger")


@dash.callback(
    Output("desc-upload-filename", "children"),
    Input("desc-gene-set-upload", "filename"),
)
def show_uploaded_filename(filename):
    if filename:
        return f"Uploaded file: {filename}"
    return ""


@dash.callback(
    Output("desc-gene-set-plot", "figure"),
    Output("desc-gene-set-summary", "children"),
    Output("desc-gene-set-table-wrapper", "children"),
    Input("desc-update-gene-set-button", "n_clicks"),
    State("desc-gene-set-file", "value"),
    State("desc-gene-set-text", "value"),
    State("desc-gene-set-upload", "contents"),
    State("desc-gene-set-upload", "filename"),
)
def update_gene_set_plots(
    n_clicks,
    gene_set_file,
    manual_text,
    upload_contents,
    upload_filename,
):
    try:
        gene_ids, missing, gene_set_label = get_gene_set_inputs(
            gene_set_file=gene_set_file,
            manual_text=manual_text,
            upload_contents=upload_contents,
            upload_filename=upload_filename,
        )

        if len(gene_ids) == 0:
            fig = go.Figure()
            fig.update_layout(
                template="plotly_white",
                title="No valid genes found",
                annotations=[
                    dict(
                        text="No valid genes were found. Please check gene IDs or gene names.",
                        x=0.5,
                        y=0.5,
                        showarrow=False,
                    )
                ],
            )

            return fig, dbc.Alert(
                "No valid genes found. Please type valid Gene IDs or GeneNames, or upload a valid gene list.",
                color="warning",
            ), ""

        fig, summary, table = make_gene_set_descriptive_figure(gene_ids, gene_set_label)

        found_labels = [f"{get_gene_name(g)} | {g}" for g in gene_ids]

        input_summary = dbc.Alert(
            [
                html.Strong("Input gene-set summary"),
                html.Br(),
                f"Resolved genes: {len(gene_ids)}",
                html.Br(),
                f"Missing or unrecognized input terms: {len(missing)}",
                html.Br(),
                html.Strong("Found genes: "),
                html.Span(", ".join(found_labels[:30]) + (" ..." if len(found_labels) > 30 else "")),
                html.Br() if missing else "",
                html.Strong("Missing terms: ") if missing else "",
                html.Span(", ".join(missing[:30]) + (" ..." if len(missing) > 30 else "")) if missing else "",
            ],
            color="secondary",
            className="mb-3",
        )

        table_show = table.copy()
        table_show["Beta"] = pd.to_numeric(table_show["Beta"], errors="coerce").round(4)

        table_component = html.Div(
            [
                html.H5("Gene-set long-format fitness table"),

                dash_table.DataTable(
                    data=table_show.to_dict("records"),
                    columns=[{"name": c, "id": c} for c in table_show.columns],
                    page_size=15,
                    filter_action="native",
                    sort_action="native",
                    style_table={
                        "overflowX": "auto",
                        "maxHeight": "520px",
                        "overflowY": "auto",
                    },
                    style_cell={
                        "textAlign": "left",
                        "fontFamily": "Arial",
                        "fontSize": "13px",
                        "padding": "6px",
                        "minWidth": "100px",
                        "whiteSpace": "normal",
                    },
                    style_header={
                        "fontWeight": "bold",
                        "backgroundColor": "#f1f3f5",
                    },
                ),

                html.Div(
                    [
                        dbc.Button(
                            "Download gene-set fitness table",
                            id="desc-download-gene-set-table-button",
                            color="secondary",
                            className="mt-3 mb-5",
                        ),
                    ]
                ),
            ],
            className="mb-5",
        )

        return fig, html.Div([input_summary, summary]), table_component

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
                    showarrow=False,
                )
            ],
        )

        return fig, dbc.Alert(str(e), color="danger"), ""


@dash.callback(
    Output("desc-download-gene-set-table", "data"),
    Input("desc-download-gene-set-table-button", "n_clicks"),
    State("desc-gene-set-file", "value"),
    State("desc-gene-set-text", "value"),
    State("desc-gene-set-upload", "contents"),
    State("desc-gene-set-upload", "filename"),
    prevent_initial_call=True,
)
def download_gene_set_table(
    n_clicks,
    gene_set_file,
    manual_text,
    upload_contents,
    upload_filename,
):
    if not n_clicks:
        return no_update

    gene_ids, missing, gene_set_label = get_gene_set_inputs(
        gene_set_file=gene_set_file,
        manual_text=manual_text,
        upload_contents=upload_contents,
        upload_filename=upload_filename,
    )

    if len(gene_ids) == 0:
        return no_update

    fig, summary, table = make_gene_set_descriptive_figure(gene_ids, gene_set_label)

    table_out = table.copy()
    table_out["Beta"] = pd.to_numeric(table_out["Beta"], errors="coerce").round(6)

    safe_label = (
        gene_set_label
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
    )

    filename = f"descriptive_fitness_{safe_label}.csv"

    return dcc.send_data_frame(table_out.to_csv, filename, index=False)
