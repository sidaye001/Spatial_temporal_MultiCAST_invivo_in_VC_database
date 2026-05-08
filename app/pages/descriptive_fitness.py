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
# Negative Beta = blue, positive Beta = brown/red.
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


def safe_file_label(label):
    return (
        str(label)
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace("|", "_")
        .replace(",", "_")
    )


def validate_time_space_selection(selected_times, selected_spaces):
    if not selected_times:
        selected_times = TIME_LEVELS.copy()
    if not selected_spaces:
        selected_spaces = SPACE_LEVELS.copy()

    selected_times = [t for t in selected_times if t in TIME_LEVELS]
    selected_spaces = [s for s in selected_spaces if s in SPACE_LEVELS]

    selected_times = [t for t in TIME_LEVELS if t in selected_times]
    selected_spaces = [s for s in SPACE_LEVELS if s in selected_spaces]

    if len(selected_times) == 0:
        selected_times = TIME_LEVELS.copy()
    if len(selected_spaces) == 0:
        selected_spaces = SPACE_LEVELS.copy()

    return selected_times, selected_spaces


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


def make_gene_set_plot_label(gene_name, gene_id):
    """
    Make a unique y-axis label for Module 2 plots.
    Plotly collapses heatmap rows when categorical y labels are duplicated.
    Using GeneName | Gene prevents genes with the same symbol/name from being overplotted.
    """
    gene_name = str(gene_name).strip() if pd.notna(gene_name) else ""
    gene_id = str(gene_id).strip()

    if gene_name == "" or gene_name == gene_id:
        return gene_id

    return f"{gene_name} | {gene_id}"


# ============================================================
# 3. Single gene spatial-temporal plot and table
# ============================================================

def get_single_gene_landscape_table(gene_id, selected_times=None, selected_spaces=None):
    selected_times, selected_spaces = validate_time_space_selection(
        selected_times,
        selected_spaces,
    )

    raw = load_raw_data()
    sub = raw[raw["Gene"] == gene_id].copy()

    if sub.empty:
        raise ValueError(f"Gene not found in raw data: {gene_id}")

    sub = sub[sub["Time"].astype(str).isin(selected_times)].copy()
    sub = sub[sub["Space"].astype(str).isin(selected_spaces)].copy()

    if sub.empty:
        raise ValueError(
            f"Gene {gene_id} was found, but no Beta values were found in the selected time/space window."
        )

    sub = add_annotation(sub)

    sub["Time"] = pd.Categorical(
        sub["Time"].astype(str),
        categories=selected_times,
        ordered=True,
    )
    sub["Space"] = pd.Categorical(
        sub["Space"].astype(str),
        categories=selected_spaces,
        ordered=True,
    )
    sub["Condition"] = sub["Time"].astype(str) + "_" + sub["Space"].astype(str)

    out = (
        sub[["Gene", "GeneName", "Time", "Space", "Condition", "Beta"]]
        .sort_values(["Time", "Space"])
        .copy()
    )

    out["Time"] = out["Time"].astype(str)
    out["Space"] = out["Space"].astype(str)

    return out


def summarize_single_gene_landscape(gene_id, selected_times=None, selected_spaces=None):
    selected_times, selected_spaces = validate_time_space_selection(
        selected_times,
        selected_spaces,
    )

    table = get_single_gene_landscape_table(
        gene_id,
        selected_times=selected_times,
        selected_spaces=selected_spaces,
    )

    finite = table[np.isfinite(pd.to_numeric(table["Beta"], errors="coerce"))].copy()
    finite["Beta"] = pd.to_numeric(finite["Beta"], errors="coerce")

    if finite.empty:
        raise ValueError(f"No finite Beta values found for gene {gene_id} in the selected window.")

    gene_name = finite["GeneName"].iloc[0]
    display_label = f"{gene_name} | {gene_id}"

    max_row = finite.loc[finite["Beta"].idxmax()]
    min_row = finite.loc[finite["Beta"].idxmin()]

    time_summary = (
        finite
        .groupby("Time", observed=False)["Beta"]
        .agg(["mean", "min", "max"])
        .reindex(selected_times)
        .reset_index()
    )

    space_summary = (
        finite
        .groupby("Space", observed=False)["Beta"]
        .agg(["mean", "min", "max"])
        .reindex(selected_spaces)
        .reset_index()
    )

    global_summary = pd.DataFrame(
        [
            {
                "Gene": gene_id,
                "GeneName": gene_name,
                "SelectedTimes": ", ".join(selected_times),
                "SelectedSpaces": ", ".join(selected_spaces),
                "n_conditions": int(finite.shape[0]),
                "mean_Beta": finite["Beta"].mean(),
                "median_Beta": finite["Beta"].median(),
                "sd_Beta": finite["Beta"].std(),
                "min_Beta": finite["Beta"].min(),
                "min_Time": min_row["Time"],
                "min_Space": min_row["Space"],
                "max_Beta": finite["Beta"].max(),
                "max_Time": max_row["Time"],
                "max_Space": max_row["Space"],
                "range_Beta": finite["Beta"].max() - finite["Beta"].min(),
                "n_positive_Beta": int((finite["Beta"] > 0).sum()),
                "n_negative_Beta": int((finite["Beta"] < 0).sum()),
            }
        ]
    )

    return {
        "table": table,
        "finite": finite,
        "display_label": display_label,
        "max_row": max_row,
        "min_row": min_row,
        "time_summary": time_summary,
        "space_summary": space_summary,
        "global_summary": global_summary,
        "selected_times": selected_times,
        "selected_spaces": selected_spaces,
    }


def make_single_gene_surface_contour(
    gene_id,
    show_contour_lines="show",
    selected_times=None,
    selected_spaces=None,
):
    selected_times, selected_spaces = validate_time_space_selection(
        selected_times,
        selected_spaces,
    )

    landscape = summarize_single_gene_landscape(
        gene_id,
        selected_times=selected_times,
        selected_spaces=selected_spaces,
    )
    sub = landscape["finite"].copy()
    display_label = landscape["display_label"]
    max_row = landscape["max_row"]
    min_row = landscape["min_row"]

    pivot = (
        sub
        .pivot_table(
            index="Space",
            columns="Time",
            values="Beta",
            aggfunc="mean",
            observed=False,
        )
        .reindex(index=selected_spaces, columns=selected_times)
    )

    z = pivot.values.astype(float)

    if np.isnan(z).all():
        raise ValueError(f"No finite Beta values found for gene {gene_id} in the selected window.")

    z_plot = pd.DataFrame(z).interpolate(axis=0).interpolate(axis=1).fillna(0).values

    max_time = str(max_row["Time"])
    max_space = str(max_row["Space"])
    max_beta = float(max_row["Beta"])

    min_time = str(min_row["Time"])
    min_space = str(min_row["Space"])
    min_beta = float(min_row["Beta"])

    selected_time_nums = [TIME_NUM_MAP[t] for t in selected_times]
    selected_space_nums = [SPACE_NUM_MAP[s] for s in selected_spaces]

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "surface"}, {"type": "xy"}]],
        subplot_titles=[
            f"3D Fitness Surface: {display_label}",
            f"Contour Map: {display_label}",
        ],
        horizontal_spacing=0.10,
    )

    fig.add_trace(
        go.Surface(
            x=selected_time_nums,
            y=selected_space_nums,
            z=z_plot,
            colorscale=NARROW_WHITE_BROWN_BLUE,
            colorbar=dict(
                title="Beta",
                x=0.44,
                y=0.50,
                len=0.68,
                thickness=14,
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
            name="3D surface",
            showscale=True,
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Heatmap(
            x=selected_times,
            y=selected_spaces,
            z=z_plot,
            colorscale=NARROW_WHITE_BROWN_BLUE,
            colorbar=dict(
                title="Beta",
                x=1.065,
                y=0.50,
                len=0.68,
                thickness=14,
            ),
            hovertemplate=(
                "Time: %{x}<br>"
                "Space: %{y}<br>"
                "Beta: %{z:.3f}<extra></extra>"
            ),
            name="Beta heatmap",
            showscale=True,
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    if show_contour_lines == "show":
        fig.add_trace(
            go.Contour(
                x=selected_times,
                y=selected_spaces,
                z=z_plot,
                contours=dict(
                    coloring="none",
                    showlabels=False,
                ),
                line=dict(color="black", width=1.2),
                showscale=False,
                hoverinfo="skip",
                name="Contour lines",
                showlegend=False,
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
            textfont=dict(color="yellow", size=13),
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
            textfont=dict(color="lime", size=13),
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
            tickvals=selected_time_nums,
            ticktext=selected_times,
        ),
        yaxis=dict(
            title="Space",
            tickmode="array",
            tickvals=selected_space_nums,
            ticktext=selected_spaces,
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
        categoryarray=selected_spaces,
        row=1,
        col=2,
    )

    fig.update_layout(
        template="plotly_white",
        height=720,
        margin=dict(l=30, r=120, t=90, b=40),
        title=f"Spatial-temporal in vivo fitness landscape for {display_label}",
        showlegend=False,
    )

    summary = dbc.Alert(
        [
            html.Strong("Single-gene fitness landscape summary"),
            html.Br(),
            f"Query gene: {display_label}",
            html.Br(),
            f"Selected timepoints: {', '.join(selected_times)}",
            html.Br(),
            f"Selected spaces: {', '.join(selected_spaces)}",
            html.Br(),
            f"Maximum Beta: {max_beta:.3f} at {max_time}, {max_space}",
            html.Br(),
            f"Minimum Beta: {min_beta:.3f} at {min_time}, {min_space}",
            html.Br(),
            f"Mean Beta across selected conditions: {landscape['global_summary']['mean_Beta'].iloc[0]:.3f}",
            html.Br(),
            html.Span(
                "The 3D surface and 2D contour map show how the fitness coefficient changes "
                "across the selected infection time and gastrointestinal (GI) tract-location window. The downloadable "
                "table below reports the full long-format single-gene landscape and summary statistics."
            ),
        ],
        color="info",
        className="mb-3",
    )

    return fig, summary, landscape


def make_single_gene_table_component(landscape):
    long_table = landscape["table"].copy()
    long_table["Beta"] = pd.to_numeric(long_table["Beta"], errors="coerce").round(4)

    global_summary = landscape["global_summary"].copy()
    for col in ["mean_Beta", "median_Beta", "sd_Beta", "min_Beta", "max_Beta", "range_Beta"]:
        global_summary[col] = pd.to_numeric(global_summary[col], errors="coerce").round(4)

    time_summary = landscape["time_summary"].copy()
    space_summary = landscape["space_summary"].copy()

    for df in [time_summary, space_summary]:
        for col in ["mean", "min", "max"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").round(4)

    return html.Div(
        [
            html.H5("Single-gene landscape summary table for selected window"),

            dash_table.DataTable(
                data=global_summary.to_dict("records"),
                columns=[{"name": c, "id": c} for c in global_summary.columns],
                page_size=5,
                style_table={"overflowX": "auto"},
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

            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H6("Time-window summary", className="mt-4"),
                            dash_table.DataTable(
                                data=time_summary.to_dict("records"),
                                columns=[{"name": c, "id": c} for c in time_summary.columns],
                                page_size=10,
                                style_table={"overflowX": "auto"},
                                style_cell={
                                    "textAlign": "left",
                                    "fontFamily": "Arial",
                                    "fontSize": "13px",
                                    "padding": "6px",
                                    "minWidth": "80px",
                                },
                                style_header={
                                    "fontWeight": "bold",
                                    "backgroundColor": "#f1f3f5",
                                },
                            ),
                        ],
                        md=6,
                    ),
                    dbc.Col(
                        [
                            html.H6("Space-window summary", className="mt-4"),
                            dash_table.DataTable(
                                data=space_summary.to_dict("records"),
                                columns=[{"name": c, "id": c} for c in space_summary.columns],
                                page_size=15,
                                style_table={"overflowX": "auto"},
                                style_cell={
                                    "textAlign": "left",
                                    "fontFamily": "Arial",
                                    "fontSize": "13px",
                                    "padding": "6px",
                                    "minWidth": "80px",
                                },
                                style_header={
                                    "fontWeight": "bold",
                                    "backgroundColor": "#f1f3f5",
                                },
                            ),
                        ],
                        md=6,
                    ),
                ]
            ),

            # The condition-level single-gene table is intentionally not displayed on the page.
            # It is still included in the downloadable CSV generated by make_single_gene_download_table().

            dbc.Button(
                "Download single-gene fitness landscape table",
                id="desc-download-single-gene-table-button",
                color="secondary",
                className="mt-3 mb-4",
            ),
        ],
        className="mb-4",
    )


def make_single_gene_download_table(gene_id, selected_times=None, selected_spaces=None):
    selected_times, selected_spaces = validate_time_space_selection(
        selected_times,
        selected_spaces,
    )

    landscape = summarize_single_gene_landscape(
        gene_id,
        selected_times=selected_times,
        selected_spaces=selected_spaces,
    )

    long_table = landscape["table"].copy()
    long_table["SelectedTimes"] = ", ".join(selected_times)
    long_table["SelectedSpaces"] = ", ".join(selected_spaces)
    long_table["TableType"] = "condition_level"
    long_table["Metric"] = ""
    long_table["Value"] = ""

    global_summary = landscape["global_summary"].copy()
    global_long = global_summary.melt(
        id_vars=["Gene", "GeneName", "SelectedTimes", "SelectedSpaces"],
        var_name="Metric",
        value_name="Value",
    )
    global_long["Time"] = ""
    global_long["Space"] = ""
    global_long["Condition"] = ""
    global_long["Beta"] = ""
    global_long["TableType"] = "global_summary"

    time_summary = landscape["time_summary"].copy()
    time_long = time_summary.melt(
        id_vars=["Time"],
        var_name="Metric",
        value_name="Value",
    )
    time_long["Gene"] = gene_id
    time_long["GeneName"] = get_gene_name(gene_id)
    time_long["SelectedTimes"] = ", ".join(selected_times)
    time_long["SelectedSpaces"] = ", ".join(selected_spaces)
    time_long["Space"] = ""
    time_long["Condition"] = ""
    time_long["Beta"] = ""
    time_long["TableType"] = "time_summary"

    space_summary = landscape["space_summary"].copy()
    space_long = space_summary.melt(
        id_vars=["Space"],
        var_name="Metric",
        value_name="Value",
    )
    space_long["Gene"] = gene_id
    space_long["GeneName"] = get_gene_name(gene_id)
    space_long["SelectedTimes"] = ", ".join(selected_times)
    space_long["SelectedSpaces"] = ", ".join(selected_spaces)
    space_long["Time"] = ""
    space_long["Condition"] = ""
    space_long["Beta"] = ""
    space_long["TableType"] = "space_summary"

    cols = [
        "TableType",
        "Gene",
        "GeneName",
        "Time",
        "Space",
        "Condition",
        "Beta",
        "Metric",
        "Value",
        "SelectedTimes",
        "SelectedSpaces",
    ]

    out = pd.concat(
        [
            long_table[cols],
            global_long[cols],
            time_long[cols],
            space_long[cols],
        ],
        ignore_index=True,
    )

    return out

# ============================================================
# 4. Gene set descriptive plots
# ============================================================

def get_gene_set_matrix(gene_ids, selected_times=None, selected_spaces=None):
    selected_times, selected_spaces = validate_time_space_selection(selected_times, selected_spaces)
    condition_order = [f"{t}_{s}" for t in selected_times for s in selected_spaces]

    raw = load_raw_data()
    sub = raw[raw["Gene"].isin(gene_ids)].copy()
    sub = sub[sub["Time"].astype(str).isin(selected_times)]
    sub = sub[sub["Space"].astype(str).isin(selected_spaces)]

    if sub.empty:
        raise ValueError(
            "None of the selected genes had data in the selected time/space window."
        )

    sub = add_annotation(sub)

    sub["Time"] = pd.Categorical(sub["Time"].astype(str), categories=selected_times, ordered=True)
    sub["Space"] = pd.Categorical(sub["Space"].astype(str), categories=selected_spaces, ordered=True)
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
        .reindex(index=ordered_genes, columns=condition_order)
    )

    label_map = {
        row["Gene"]: make_gene_set_plot_label(row["GeneName"], row["Gene"])
        for _, row in gene_order.iterrows()
    }
    y_labels = [label_map.get(g, g) for g in mat.index]

    return sub, mat, y_labels, selected_times, selected_spaces, condition_order


def make_gene_set_descriptive_figure(
    gene_ids,
    gene_set_label,
    selected_times=None,
    selected_spaces=None,
):
    sub, mat, y_labels, selected_times, selected_spaces, condition_order = get_gene_set_matrix(
        gene_ids,
        selected_times=selected_times,
        selected_spaces=selected_spaces,
    )

    n_genes = sub["Gene"].nunique()

    temporal = (
        sub
        .groupby("Time", observed=False)["Beta"]
        .agg(["mean", "sem"])
        .reindex(selected_times)
        .reset_index()
    )
    temporal["sem"] = temporal["sem"].fillna(0)

    spatial = (
        sub
        .groupby("Space", observed=False)["Beta"]
        .agg(["mean", "sem"])
        .reindex(selected_spaces)
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
        .reindex(index=selected_times, columns=selected_spaces)
    )

    ranked = (
        sub
        .groupby(["Gene", "GeneName"], observed=False)["Beta"]
        .mean()
        .reset_index()
        .sort_values("Beta")
    )
    ranked["Label"] = ranked.apply(
        lambda r: make_gene_set_plot_label(r["GeneName"], r["Gene"]),
        axis=1,
    )

    heatmap_x_labels = [c.replace("_", "<br>") for c in condition_order]
    heatmap_customdata = np.tile(np.array(condition_order, dtype=object), (len(y_labels), 1))

    fig = make_subplots(
        rows=3,
        cols=3,
        specs=[
            [{"type": "heatmap", "colspan": 3}, None, None],
            [{"type": "xy"}, {"type": "xy"}, {"type": "heatmap"}],
            [{"type": "xy", "colspan": 2}, None, {"type": "xy"}],
        ],
        subplot_titles=[
            f"{gene_set_label}: gene-level fitness heatmap",
            "Mean Beta across selected timepoints",
            "Mean Beta across selected GI tract locations",
            "Mean Beta by time × GI tract location",
            "Genes ranked by mean Beta",
            "Beta distribution by selected GI tract location",
        ],
        row_heights=[0.48, 0.24, 0.28],
        vertical_spacing=0.09,
        horizontal_spacing=0.095,
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
            x=heatmap_x_labels,
            y=y_labels,
            customdata=heatmap_customdata,
            colorscale=NARROW_WHITE_BROWN_BLUE,
            zmin=zmin,
            zmax=zmax,
            zmid=0,
            xgap=0.4,
            ygap=0.6,
            colorbar=dict(
                title="Beta",
                x=1.015,
                y=0.88,
                len=0.26,
                thickness=13,
            ),
            hovertemplate=(
                "Gene: %{y}<br>"
                "Condition: %{customdata}<br>"
                "Beta: %{z:.3f}<extra></extra>"
            ),
            showlegend=False,
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
            x=selected_spaces,
            y=selected_times,
            colorscale=NARROW_WHITE_BROWN_BLUE,
            zmid=0,
            colorbar=dict(
                title="Mean Beta",
                x=1.015,
                # Align the Mean Beta colorbar with the second-row interaction heatmap
                # so it does not overlap the first-row gene-level heatmap.
                y=0.455,
                len=0.16,
                thickness=13,
            ),
            hovertemplate=(
                "Space: %{x}<br>"
                "Time: %{y}<br>"
                "Mean Beta: %{z:.3f}<extra></extra>"
            ),
            showlegend=False,
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

    for s in selected_spaces:
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
            row=3,
            col=3,
        )

    fig.update_xaxes(
        tickangle=0,
        title_text="Condition",
        tickfont=dict(size=9),
        automargin=True,
        row=1,
        col=1,
    )
    fig.update_yaxes(
        title_text="Gene",
        tickfont=dict(size=11),
        automargin=True,
        row=1,
        col=1,
    )

    fig.update_xaxes(title_text="Timepoint", row=2, col=1)
    fig.update_yaxes(title_text="Mean Beta coefficient", row=2, col=1)

    fig.update_xaxes(title_text="Gastrointestinal (GI) tract location", tickangle=45, row=2, col=2)
    fig.update_yaxes(title_text="Mean Beta coefficient", row=2, col=2)

    fig.update_xaxes(title_text="Location", tickangle=45, row=2, col=3)
    fig.update_yaxes(title_text="Timepoint", row=2, col=3)

    fig.update_xaxes(
        title_text="Mean Beta coefficient",
        zeroline=True,
        row=3,
        col=1,
    )
    fig.update_yaxes(
        title_text="Gene",
        automargin=True,
        tickfont=dict(size=10),
        row=3,
        col=1,
    )

    fig.update_xaxes(
        title_text="Gastrointestinal (GI) tract location",
        tickangle=45,
        tickfont=dict(size=10),
        row=3,
        col=3,
    )
    fig.update_yaxes(title_text="Beta coefficient", row=3, col=3)

    dynamic_height = max(1250, 900 + 24 * n_genes)

    fig.update_layout(
        title=(
            f"Descriptive fitness summary for {gene_set_label} "
            f"({n_genes} genes found; {len(selected_times)} timepoints × {len(selected_spaces)} spaces)"
        ),
        template="plotly_white",
        height=dynamic_height,
        margin=dict(l=150, r=125, t=115, b=90),
        showlegend=False,
        font=dict(size=12),
    )

    fig.update_annotations(font=dict(size=14))

    summary = dbc.Alert(
        [
            html.Strong("Gene-set summary"),
            html.Br(),
            f"Input gene set: {gene_set_label}",
            html.Br(),
            f"Number of genes found in selected window: {n_genes}",
            html.Br(),
            f"Selected time window: {', '.join(selected_times)}",
            html.Br(),
            f"Selected space window: {', '.join(selected_spaces)}",
            html.Br(),
            html.Span(
                "The top heatmap shows each gene across the selected spatial-temporal conditions. "
                "The temporal, spatial, and interaction panels summarize the average pattern "
                "within the selected window. The ranked bar plot orders genes by mean Beta "
                "coefficient inside the selected window, and the box plot shows Beta distributions "
                "across the selected gastrointestinal (GI) tract locations."
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

    gene_summary = (
        table
        .groupby(["Gene", "GeneName"], observed=False)["Beta"]
        .agg(
            mean_Beta="mean",
            median_Beta="median",
            sd_Beta="std",
            min_Beta="min",
            max_Beta="max",
            n_conditions="count",
        )
        .reset_index()
        .sort_values(["mean_Beta", "GeneName", "Gene"])
    )

    return fig, summary, table, gene_summary


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
                    "Use the first module to inspect one gene across infection time and gastrointestinal (GI) tract location. "
                    "Use the second module to summarize a predefined or uploaded gene set across selected "
                    "time and space windows. Beta represents the fitness coefficient used in the dataset."
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
                dbc.Col(
                    [
                        html.Label("Contour line overlay"),
                        dcc.RadioItems(
                            id="desc-single-contour-lines",
                            options=[
                                {
                                    "label": "Hide contour line trace",
                                    "value": "hide",
                                },
                                {
                                    "label": "Show contour line overlay",
                                    "value": "show",
                                },
                            ],
                            value="show",
                            inline=True,
                            inputStyle={"marginRight": "6px", "marginLeft": "12px"},
                        ),
                        html.Small(
                            "Default shows contour lines. Hide it if you want to remove the extra Plotly trace.",
                            className="text-muted",
                        ),
                    ],
                    md=6,
                ),
            ],
            className="mb-3",
        ),

        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Label("Single-gene time window"),
                        dcc.Dropdown(
                            id="desc-single-gene-time-window",
                            options=[
                                {"label": t, "value": t}
                                for t in TIME_LEVELS
                            ],
                            value=TIME_LEVELS,
                            multi=True,
                            clearable=False,
                            placeholder="Choose timepoints",
                        ),
                        html.Small(
                            "Default uses all infection timepoints.",
                            className="text-muted",
                        ),
                    ],
                    md=5,
                ),
                dbc.Col(
                    [
                        html.Label("Single-gene space window"),
                        dcc.Dropdown(
                            id="desc-single-gene-space-window",
                            options=[
                                {"label": s, "value": s}
                                for s in SPACE_LEVELS
                            ],
                            value=SPACE_LEVELS,
                            multi=True,
                            clearable=False,
                            placeholder="Choose gastrointestinal (GI) tract locations",
                        ),
                        html.Small(
                            "Default uses all gastrointestinal (GI) tract locations.",
                            className="text-muted",
                        ),
                    ],
                    md=7,
                ),
            ],
            className="mb-3",
        ),

        dcc.Loading(
            type="circle",
            children=[
                html.Div(id="desc-single-gene-summary"),
                dcc.Graph(id="desc-single-gene-plot", style={"width": "100%"}),
                html.Div(id="desc-single-gene-table-wrapper"),
            ],
        ),

        html.Hr(),

        html.H4("2. Gene-set descriptive fitness summary"),

        dbc.Alert(
            [
                html.Strong("Input options: "),
                html.Span(
                    "Choose any predefined gene set from ../data/gene_sets, paste gene IDs or gene names, "
                    "or upload a CSV/TXT/TSV file. Manual input and uploaded genes are added to the predefined set. "
                    "Use the time and space filters to summarize only a selected window."
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
                    [
                        html.Label("Time window"),
                        dcc.Dropdown(
                            id="desc-gene-set-time-window",
                            options=[{"label": t, "value": t} for t in TIME_LEVELS],
                            value=TIME_LEVELS,
                            multi=True,
                            clearable=False,
                            placeholder="Choose one or more timepoints",
                        ),
                        html.Small(
                            "Default: all timepoints. The selected order follows the original biological order.",
                            className="text-muted",
                        ),
                    ],
                    md=6,
                ),
                dbc.Col(
                    [
                        html.Label("Space window"),
                        dcc.Dropdown(
                            id="desc-gene-set-space-window",
                            options=[{"label": s, "value": s} for s in SPACE_LEVELS],
                            value=SPACE_LEVELS,
                            multi=True,
                            clearable=False,
                            placeholder="Choose one or more gastrointestinal (GI) tract locations",
                        ),
                        html.Small(
                            "Default: all spaces. You can restrict to proximal, distal, cecum/colon, etc.",
                            className="text-muted",
                        ),
                    ],
                    md=6,
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

        dcc.Download(id="desc-download-single-gene-table"),
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
    Output("desc-single-gene-table-wrapper", "children"),
    Input("desc-single-gene", "value"),
    Input("desc-single-contour-lines", "value"),
    Input("desc-single-gene-time-window", "value"),
    Input("desc-single-gene-space-window", "value"),
)
def update_single_gene_plot(gene_id, show_contour_lines, selected_times, selected_spaces):
    try:
        if gene_id is None:
            fig = go.Figure()
            fig.update_layout(template="plotly_white", title="No gene selected")
            return fig, dbc.Alert("No gene selected.", color="warning"), ""

        selected_times, selected_spaces = validate_time_space_selection(
            selected_times,
            selected_spaces,
        )

        fig, summary, landscape = make_single_gene_surface_contour(
            gene_id,
            show_contour_lines=show_contour_lines,
            selected_times=selected_times,
            selected_spaces=selected_spaces,
        )
        table_component = make_single_gene_table_component(landscape)

        return fig, summary, table_component

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
    State("desc-gene-set-time-window", "value"),
    State("desc-gene-set-space-window", "value"),
)
def update_gene_set_plots(
    n_clicks,
    gene_set_file,
    manual_text,
    upload_contents,
    upload_filename,
    selected_times,
    selected_spaces,
):
    try:
        selected_times, selected_spaces = validate_time_space_selection(
            selected_times,
            selected_spaces,
        )

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

        fig, summary, table, gene_summary = make_gene_set_descriptive_figure(
            gene_ids,
            gene_set_label,
            selected_times=selected_times,
            selected_spaces=selected_spaces,
        )

        found_labels = [f"{get_gene_name(g)} | {g}" for g in gene_ids]

        input_summary = dbc.Alert(
            [
                html.Strong("Input gene-set summary"),
                html.Br(),
                f"Resolved genes: {len(gene_ids)}",
                html.Br(),
                f"Missing or unrecognized input terms: {len(missing)}",
                html.Br(),
                f"Selected timepoints: {', '.join(selected_times)}",
                html.Br(),
                f"Selected spaces: {', '.join(selected_spaces)}",
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

        # Module 2 tables are intentionally not displayed on the page.
        # The downloadable CSV still includes both:
        #   1) condition-level long-format fitness table
        #   2) per-gene summary table for the selected window
        table_component = html.Div(
            [
                dbc.Alert(
                    "Gene-set condition-level and per-gene summary tables are available in the download file.",
                    color="light",
                    className="mb-2",
                ),
                dbc.Button(
                    "Download gene-set fitness table",
                    id="desc-download-gene-set-table-button",
                    color="secondary",
                    className="mt-2 mb-5",
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
    Output("desc-download-single-gene-table", "data"),
    Input("desc-download-single-gene-table-button", "n_clicks"),
    State("desc-single-gene", "value"),
    State("desc-single-gene-time-window", "value"),
    State("desc-single-gene-space-window", "value"),
    prevent_initial_call=True,
)
def download_single_gene_table(n_clicks, gene_id, selected_times, selected_spaces):
    if not n_clicks or gene_id is None:
        return no_update

    selected_times, selected_spaces = validate_time_space_selection(
        selected_times,
        selected_spaces,
    )

    table_out = make_single_gene_download_table(
        gene_id,
        selected_times=selected_times,
        selected_spaces=selected_spaces,
    )
    table_out["Beta"] = pd.to_numeric(table_out["Beta"], errors="coerce").round(6)

    filename = (
        f"single_gene_fitness_landscape_{safe_file_label(get_gene_name(gene_id))}_{safe_file_label(gene_id)}"
        f"_time-{safe_file_label('-'.join(selected_times))}"
        f"_space-{safe_file_label('-'.join(selected_spaces))}.csv"
    )

    return dcc.send_data_frame(table_out.to_csv, filename, index=False)


@dash.callback(
    Output("desc-download-gene-set-table", "data"),
    Input("desc-download-gene-set-table-button", "n_clicks"),
    State("desc-gene-set-file", "value"),
    State("desc-gene-set-text", "value"),
    State("desc-gene-set-upload", "contents"),
    State("desc-gene-set-upload", "filename"),
    State("desc-gene-set-time-window", "value"),
    State("desc-gene-set-space-window", "value"),
    prevent_initial_call=True,
)
def download_gene_set_table(
    n_clicks,
    gene_set_file,
    manual_text,
    upload_contents,
    upload_filename,
    selected_times,
    selected_spaces,
):
    if not n_clicks:
        return no_update

    selected_times, selected_spaces = validate_time_space_selection(
        selected_times,
        selected_spaces,
    )

    gene_ids, missing, gene_set_label = get_gene_set_inputs(
        gene_set_file=gene_set_file,
        manual_text=manual_text,
        upload_contents=upload_contents,
        upload_filename=upload_filename,
    )

    if len(gene_ids) == 0:
        return no_update

    fig, summary, table, gene_summary = make_gene_set_descriptive_figure(
        gene_ids,
        gene_set_label,
        selected_times=selected_times,
        selected_spaces=selected_spaces,
    )

    table_out = table.copy()
    table_out["Beta"] = pd.to_numeric(table_out["Beta"], errors="coerce").round(6)
    table_out["SelectedTimes"] = ", ".join(selected_times)
    table_out["SelectedSpaces"] = ", ".join(selected_spaces)

    gene_summary_out = gene_summary.copy()
    gene_summary_out["Time"] = ""
    gene_summary_out["Space"] = ""
    gene_summary_out["Condition"] = ""
    gene_summary_out["Beta"] = ""
    gene_summary_out["SelectedTimes"] = ", ".join(selected_times)
    gene_summary_out["SelectedSpaces"] = ", ".join(selected_spaces)

    table_out["TableType"] = "condition_level"
    gene_summary_out["TableType"] = "gene_summary"

    all_cols = [
        "TableType",
        "Gene",
        "GeneName",
        "Time",
        "Space",
        "Condition",
        "Beta",
        "mean_Beta",
        "median_Beta",
        "sd_Beta",
        "min_Beta",
        "max_Beta",
        "n_conditions",
        "SelectedTimes",
        "SelectedSpaces",
    ]

    for col in all_cols:
        if col not in table_out.columns:
            table_out[col] = ""
        if col not in gene_summary_out.columns:
            gene_summary_out[col] = ""

    combined = pd.concat(
        [table_out[all_cols], gene_summary_out[all_cols]],
        ignore_index=True,
    )

    filename = (
        f"descriptive_fitness_{safe_file_label(gene_set_label)}"
        f"_time-{safe_file_label('-'.join(selected_times))}"
        f"_space-{safe_file_label('-'.join(selected_spaces))}.csv"
    )

    return dcc.send_data_frame(combined.to_csv, filename, index=False)
