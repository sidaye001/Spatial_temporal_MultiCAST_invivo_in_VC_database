import os
import base64
import io
from functools import lru_cache

import numpy as np
import pandas as pd
import networkx as nx
import dash
from dash import html, dcc, dash_table, Input, Output, State, no_update
import dash_bootstrap_components as dbc
import plotly.graph_objects as go


dash.register_page(__name__, path="/network-browser", name="Network Browser")


# ============================================================
# 0. Paths
# ============================================================

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(APP_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

NETWORK_DIR = os.path.join(DATA_DIR, "network")
GENE_SET_DIR = os.path.join(DATA_DIR, "gene_sets")

EDGE_FILE = os.path.join(
    NETWORK_DIR,
    "GGM_edge_table_for_cytoscape.csv"
)

NODE_FILE = os.path.join(
    NETWORK_DIR,
    "GGM_node_table_for_cytoscape.csv"
)

NODE_FINAL_FILE = os.path.join(
    NETWORK_DIR,
    "GGM_node_table_hub_gene_rank.csv"
)

ANNOTATION_FILE = os.path.join(
    DATA_DIR,
    "annotation",
    "new_annotations_with_uniprot_names.csv"
)

GENE_SET_FILES = {
    "Flagella": os.path.join(GENE_SET_DIR, "Flagella_genes.csv"),
    "O_antigen": os.path.join(GENE_SET_DIR, "O_antigen_genes.csv"),
    "Tcp": os.path.join(GENE_SET_DIR, "Tcp_genes.csv"),
    # Add more predefined gene sets here:
    # "Bile_resistance": os.path.join(GENE_SET_DIR, "bile_resistance_genes.csv"),
    # "Biotin": os.path.join(GENE_SET_DIR, "biotin_genes.csv"),
}

DEFAULT_QUERY = "Tcp"
DEFAULT_QUERY_COLOR = "#E7298A"


# ============================================================
# 1. Data loading
# ============================================================

@lru_cache(maxsize=1)
def load_edges():
    if not os.path.exists(EDGE_FILE):
        raise FileNotFoundError(f"Network edge file not found: {EDGE_FILE}")

    df = pd.read_csv(EDGE_FILE)

    required = {"source", "target"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in edge table: {missing}")

    df["source"] = df["source"].astype(str)
    df["target"] = df["target"].astype(str)

    if "partial_corr" not in df.columns:
        if "weight" in df.columns:
            df["partial_corr"] = pd.to_numeric(df["weight"], errors="coerce")
        else:
            df["partial_corr"] = np.nan

    df["partial_corr"] = pd.to_numeric(df["partial_corr"], errors="coerce")

    if "abs_partial_corr" not in df.columns:
        df["abs_partial_corr"] = df["partial_corr"].abs()

    df["abs_partial_corr"] = pd.to_numeric(df["abs_partial_corr"], errors="coerce")

    if "edge_type" not in df.columns:
        df["edge_type"] = np.where(df["partial_corr"] >= 0, "positive", "negative")

    df["edge_type"] = df["edge_type"].fillna("unknown").astype(str)

    return df


@lru_cache(maxsize=1)
def load_nodes():
    if not os.path.exists(NODE_FILE):
        raise FileNotFoundError(f"Network node file not found: {NODE_FILE}")

    df = pd.read_csv(NODE_FILE)

    if "id" not in df.columns:
        first_col = df.columns[0]
        df = df.rename(columns={first_col: "id"})

    if "degree" not in df.columns:
        if df.shape[1] >= 2:
            second_col = df.columns[1]
            df = df.rename(columns={second_col: "degree"})
        else:
            df["degree"] = 1

    df["id"] = df["id"].astype(str)
    df["degree"] = pd.to_numeric(df["degree"], errors="coerce").fillna(1)

    return df


@lru_cache(maxsize=1)
def load_node_final():
    if os.path.exists(NODE_FINAL_FILE):
        df = pd.read_csv(NODE_FINAL_FILE)
    else:
        base = load_nodes().copy()
        df = base.rename(columns={"id": "Gene"})

    if "Gene" not in df.columns:
        first_col = df.columns[0]
        df = df.rename(columns={first_col: "Gene"})

    df["Gene"] = df["Gene"].astype(str)

    if "degree" not in df.columns:
        df["degree"] = 1

    df["degree"] = pd.to_numeric(df["degree"], errors="coerce").fillna(1)

    return df


@lru_cache(maxsize=1)
def load_annotation():
    if not os.path.exists(ANNOTATION_FILE):
        nodes = load_nodes()
        return pd.DataFrame({
            "Gene": nodes["id"].astype(str),
            "GeneName": nodes["id"].astype(str),
            "VC_ID": "",
        })

    ann = pd.read_csv(ANNOTATION_FILE)

    if "locus_ID" not in ann.columns:
        nodes = load_nodes()
        return pd.DataFrame({
            "Gene": nodes["id"].astype(str),
            "GeneName": nodes["id"].astype(str),
            "VC_ID": "",
        })

    if "gene_name" not in ann.columns:
        ann["gene_name"] = ""
    if "KEGG_VC_number" not in ann.columns:
        ann["KEGG_VC_number"] = ""

    ann = ann[["locus_ID", "gene_name", "KEGG_VC_number"]].copy()
    ann["locus_ID"] = ann["locus_ID"].astype(str)
    ann["gene_name"] = ann["gene_name"].fillna("").astype(str)
    ann["KEGG_VC_number"] = ann["KEGG_VC_number"].fillna("").astype(str).str.strip()

    ann["GeneName"] = np.where(
        ann["gene_name"].str.strip() != "",
        ann["gene_name"],
        ann["locus_ID"]
    )

    ann = ann.rename(columns={"locus_ID": "Gene", "KEGG_VC_number": "VC_ID"})
    ann = ann[["Gene", "GeneName", "VC_ID"]].drop_duplicates()

    return ann


@lru_cache(maxsize=1)
def load_node_annotated():
    node = load_nodes().copy()
    ann = load_annotation()

    node = node.merge(
        ann,
        left_on="id",
        right_on="Gene",
        how="left"
    )

    node["GeneName"] = node["GeneName"].fillna(node["id"])
    node["VC_ID"] = node["VC_ID"].fillna("").astype(str).str.strip()

    return node


@lru_cache(maxsize=1)
def load_gene_lookup_options():
    """
    Build searchable dropdown options for query genes.

    Users can search by GeneName, Gene ID, or KEGG VC number. The dropdown value is
    the canonical network node ID/locus ID. Predefined gene sets are included
    as selectable options using encoded values.
    """
    node_anno = load_node_annotated().copy()
    node_anno["id"] = node_anno["id"].astype(str)
    node_anno["GeneName"] = node_anno["GeneName"].fillna(node_anno["id"]).astype(str)
    node_anno["VC_ID"] = node_anno["VC_ID"].fillna("").astype(str).str.strip()
    node_anno["DisplayLabel"] = node_anno["GeneName"] + " | " + node_anno["id"]
    node_anno["DisplayLabel"] = np.where(
        node_anno["VC_ID"] != "",
        node_anno["DisplayLabel"] + " | " + node_anno["VC_ID"],
        node_anno["DisplayLabel"],
    )

    gene_options = [
        {"label": row["DisplayLabel"], "value": row["id"]}
        for _, row in node_anno.sort_values(["GeneName", "id"]).iterrows()
    ]

    gene_set_options = [
        {"label": f"Gene set: {key}", "value": f"__gene_set__::{key}"}
        for key in sorted(GENE_SET_FILES.keys(), key=lambda x: x.lower())
    ]

    return gene_set_options + gene_options


def extract_gene_col(df):
    possible_cols = [
        "locus_ID", "Locus_ID",
        "Gene", "gene",
        "id", "ID",
        "X",
        "GeneID", "gene_id",
        "GeneName", "gene_name",
        "VC_ID", "KEGGVC", "KEGG_VC_number"
    ]

    hit = [c for c in possible_cols if c in df.columns]

    if len(hit) == 0:
        selected_col = df.columns[0]
    else:
        selected_col = hit[0]

    genes = (
        df[selected_col]
        .dropna()
        .astype(str)
        .str.strip()
        .tolist()
    )

    genes = [g for g in genes if g != ""]

    return list(dict.fromkeys(genes))


def parse_gene_tokens(text):
    """
    Parse manual/query input.

    Accepts either free text or a list from dcc.Dropdown(multi=True).
    """
    if text is None:
        return []

    if isinstance(text, (list, tuple, set)):
        raw_items = []
        for item in text:
            raw_items.extend(parse_gene_tokens(item))
        return list(dict.fromkeys(raw_items))

    text = str(text)

    for sep in [",", ";", "\t", "\r", "\n"]:
        text = text.replace(sep, " ")

    tokens = [
        x.strip()
        for x in text.split(" ")
        if x.strip() != ""
    ]

    bad_headers = {
        "gene", "genes", "geneid", "gene_id",
        "locus_id", "genename", "gene_name",
        "keggvc", "kegg_vc_number"
    }

    tokens = [x for x in tokens if x.lower() not in bad_headers]

    return list(dict.fromkeys(tokens))


def parse_uploaded_gene_file(contents, filename):
    if contents is None:
        return []

    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    text = decoded.decode("utf-8", errors="replace")

    filename = filename or ""

    if filename.endswith(".csv"):
        df = pd.read_csv(io.StringIO(text))
        return extract_gene_col(df)

    if filename.endswith(".tsv"):
        df = pd.read_csv(io.StringIO(text), sep="\t")
        return extract_gene_col(df)

    return parse_gene_tokens(text)


def get_predefined_gene_set_key(query_text):
    tokens = parse_gene_tokens(query_text)

    if len(tokens) != 1:
        return None

    q_raw = str(tokens[0]).strip()

    if q_raw.startswith("__gene_set__::"):
        q_raw = q_raw.split("::", 1)[1]

    q = q_raw.lower()

    for key in GENE_SET_FILES.keys():
        if q == key.lower():
            return key

    return None


def load_gene_set_by_key(key):
    path = GENE_SET_FILES.get(key)

    if path is None or not os.path.exists(path):
        raise FileNotFoundError(f"Gene set file not found for {key}: {path}")

    df = pd.read_csv(path)
    genes = extract_gene_col(df)

    return genes


def resolve_gene_ids(tokens):
    node_anno = load_node_annotated()

    valid_ids = set(node_anno["id"].astype(str))

    lookup = {}

    for _, row in node_anno.iterrows():
        gid = str(row["id"])
        gname = str(row["GeneName"])
        vc = str(row.get("VC_ID", "")).strip()

        lookup.setdefault(gid.lower(), []).append(gid)
        lookup.setdefault(gname.lower(), []).append(gid)
        if vc:
            lookup.setdefault(vc.lower(), []).append(gid)

    resolved = []
    missing = []

    for token in tokens:
        token = str(token).strip()
        token_lower = token.lower()

        if token in valid_ids:
            resolved.append(token)
        elif token_lower in lookup:
            resolved.extend(lookup[token_lower])
        else:
            missing.append(token)

    resolved = list(dict.fromkeys(resolved))

    return resolved, missing



def expand_predefined_gene_set_tokens(tokens):
    """
    Allow users to mix individual genes and predefined gene sets in the
    searchable dropdown/manual query.
    """
    expanded_tokens = []

    for token in tokens:
        token_str = str(token).strip()
        key = None

        if token_str.startswith("__gene_set__::"):
            key = token_str.split("::", 1)[1]
        else:
            for candidate in GENE_SET_FILES.keys():
                if token_str.lower() == candidate.lower():
                    key = candidate
                    break

        if key is not None and key in GENE_SET_FILES:
            expanded_tokens.extend(load_gene_set_by_key(key))
        else:
            expanded_tokens.append(token_str)

    return list(dict.fromkeys(expanded_tokens))

def get_query_genes(query_source, query_text, upload_contents, upload_filename):
    if query_source == "upload":
        tokens = parse_uploaded_gene_file(upload_contents, upload_filename)
        resolved, missing = resolve_gene_ids(tokens)

        if len(resolved) == 0:
            raise ValueError(
                "No valid query genes found in uploaded file. "
                "Please upload a CSV/TSV/TXT file with gene IDs, gene names, or KEGG VC numbers."
            )

        return resolved, missing, f"Using uploaded gene list: {upload_filename}"

    query_text = query_text or ""

    predefined_key = get_predefined_gene_set_key(query_text)

    if predefined_key is not None:
        tokens = load_gene_set_by_key(predefined_key)
        resolved, missing = resolve_gene_ids(tokens)

        if len(resolved) == 0:
            raise ValueError(f"No genes from predefined gene set {predefined_key} were found in the network.")

        return resolved, missing, f"Using predefined gene set: {predefined_key}"

    tokens = parse_gene_tokens(query_text)
    tokens = expand_predefined_gene_set_tokens(tokens)
    resolved, missing = resolve_gene_ids(tokens)

    if len(resolved) == 0:
        available_sets = ", ".join(GENE_SET_FILES.keys())
        raise ValueError(
            "No valid query genes found. "
            "Please enter a valid gene ID, gene name, KEGG VC number, or predefined gene set name. "
            f"Available predefined gene sets: {available_sets}"
        )

    return resolved, missing, "Using searched query gene(s)."


def gene_label(gene_id):
    ann = load_annotation()
    mp = dict(zip(ann["Gene"], ann["GeneName"]))
    return mp.get(gene_id, gene_id)


def normalize_color(color_value, default=DEFAULT_QUERY_COLOR):
    if color_value is None:
        return default

    color_value = str(color_value).strip()

    if color_value == "":
        return default

    return color_value


# ============================================================
# 2. Network construction and layout
# ============================================================

def build_filtered_full_network(
    query_genes,
    top_edge_number,
    include_query_edges=True,
    query_edge_mode="all"
):
    edge_table = load_edges()
    node_anno = load_node_annotated()

    edge_top = (
        edge_table
        .sort_values("abs_partial_corr", ascending=False)
        .head(int(top_edge_number))
        .copy()
    )

    if include_query_edges:
        query_edges = edge_table[
            edge_table["source"].isin(query_genes) |
            edge_table["target"].isin(query_genes)
        ].copy()

        if query_edge_mode == "query_only":
            query_edges = query_edges[
                query_edges["source"].isin(query_genes) |
                query_edges["target"].isin(query_genes)
            ].copy()

        if len(query_edges) > 0:
            query_edges = query_edges.sort_values("abs_partial_corr", ascending=False)
            query_edges = query_edges.head(min(len(query_edges), int(top_edge_number)))

        edge_filt = pd.concat([edge_top, query_edges], axis=0)
        edge_filt = edge_filt.drop_duplicates(subset=["source", "target"])
    else:
        edge_filt = edge_top.copy()

    nodes_in_edges = pd.unique(pd.concat([edge_filt["source"], edge_filt["target"]], axis=0))
    node_filt = node_anno[node_anno["id"].isin(nodes_in_edges)].copy()

    node_filt["is_query"] = node_filt["id"].isin(query_genes)
    node_filt["node_type"] = np.where(node_filt["is_query"], "Query gene", "Other")

    return edge_filt, node_filt


def graph_from_edges_nodes(edge_df, node_df):
    G = nx.Graph()

    for _, row in node_df.iterrows():
        G.add_node(
            row["id"],
            GeneName=row.get("GeneName", row["id"]),
            VC_ID=row.get("VC_ID", ""),
            degree=float(row.get("degree", 1)),
            node_type=row.get("node_type", "Other"),
            is_query=bool(row.get("is_query", False))
        )

    for _, row in edge_df.iterrows():
        source = str(row["source"])
        target = str(row["target"])

        if source in G.nodes and target in G.nodes:
            G.add_edge(
                source,
                target,
                partial_corr=float(row["partial_corr"]) if pd.notna(row["partial_corr"]) else 0.0,
                abs_partial_corr=float(row["abs_partial_corr"]) if pd.notna(row["abs_partial_corr"]) else 0.0,
                edge_type=str(row.get("edge_type", "unknown"))
            )

    return G


def compute_layout_3d_fr(G, seed=222):
    """
    3D Fruchterman-Reingold force-directed layout.

    This mirrors igraph::layout_with_fr(..., dim = 3) logic in R.
    Edge weights use |partial correlation| so stronger edges pull nodes closer.
    """
    if G.number_of_nodes() == 0:
        return {}

    pos = nx.spring_layout(
        G,
        dim=3,
        seed=int(seed),
        weight="abs_partial_corr",
        iterations=250,
        scale=1.0
    )

    return pos


def compute_layout_3d_sphere(G, seed=222):
    """
    Spherical 3D layout.

    Nodes are distributed on a sphere surface using a deterministic Fibonacci sphere.
    Higher-degree genes are placed slightly closer to the sphere center so hubs are more central.
    """
    if G.number_of_nodes() == 0:
        return {}

    nodes = list(G.nodes())
    n = len(nodes)

    rng = np.random.default_rng(int(seed))
    order = np.arange(n)
    rng.shuffle(order)

    phi = np.pi * (3.0 - np.sqrt(5.0))

    degrees = np.array([G.nodes[node].get("degree", G.degree(node)) for node in nodes], dtype=float)
    degrees = np.nan_to_num(degrees, nan=1.0, posinf=1.0, neginf=1.0)

    if np.max(degrees) > np.min(degrees):
        deg_scaled = (np.log1p(degrees) - np.min(np.log1p(degrees))) / (
            np.max(np.log1p(degrees)) - np.min(np.log1p(degrees))
        )
    else:
        deg_scaled = np.zeros_like(degrees)

    pos = {}

    for rank, node_idx in enumerate(order):
        node = nodes[node_idx]

        y = 1.0 - (rank / max(n - 1, 1)) * 2.0
        radius_at_y = np.sqrt(max(0.0, 1.0 - y * y))
        theta = phi * rank

        x = np.cos(theta) * radius_at_y
        z = np.sin(theta) * radius_at_y

        # hubs move slightly inward; low-degree nodes stay near the sphere surface
        radial_scale = 1.0 - 0.22 * deg_scaled[node_idx]

        pos[node] = np.array([x * radial_scale, y * radial_scale, z * radial_scale])

    return pos


def compute_layout_3d(G, seed=222, layout_method="fr"):
    if layout_method == "sphere":
        return compute_layout_3d_sphere(G, seed=seed)

    return compute_layout_3d_fr(G, seed=seed)


def compute_layout_2d(G, seed=123):
    """
    2D Fruchterman-Reingold force-directed layout.
    """
    if G.number_of_nodes() == 0:
        return {}

    pos = nx.spring_layout(
        G,
        dim=2,
        seed=int(seed),
        weight="abs_partial_corr",
        iterations=150,
        scale=1.0
    )

    return pos


def edge_width_from_abs_corr(abs_corr, min_width=2.2, max_width=10.5):
    """
    Convert |partial correlation| into a visible Plotly edge width.
    """
    if pd.isna(abs_corr):
        return min_width

    abs_corr = float(abs_corr)

    scaled = np.sqrt(max(abs_corr, 0.0))
    scaled = min(scaled / np.sqrt(0.30), 1.0)

    return min_width + scaled * (max_width - min_width)


def full_map_edge_width_from_abs_corr(abs_corr, min_width=2.6, max_width=8.2):
    """
    Edge-width scaling for the full 3D map.
    """
    if pd.isna(abs_corr):
        return min_width

    abs_corr = float(abs_corr)

    scaled = np.sqrt(max(abs_corr, 0.0))
    scaled = min(scaled / np.sqrt(0.30), 1.0)

    return min_width + scaled * (max_width - min_width)


def query_edge_width_from_abs_corr(abs_corr, min_width=4.0, max_width=12.0):
    """
    Extra-thick edge-width scaling for edges directly connected to query genes.
    """
    if pd.isna(abs_corr):
        return min_width

    abs_corr = float(abs_corr)

    scaled = np.sqrt(max(abs_corr, 0.0))
    scaled = min(scaled / np.sqrt(0.30), 1.0)

    return min_width + scaled * (max_width - min_width)


def node_size_from_degree(degree, min_size=8, max_size=26):
    """
    Convert full-network degree into node marker size.
    Uses log scaling so high-degree hubs are visible without dominating the plot.
    """
    try:
        degree = float(degree)
    except Exception:
        return min_size

    if not np.isfinite(degree) or degree <= 0:
        return min_size

    scaled = np.log1p(degree) / np.log1p(300)
    scaled = max(0.0, min(scaled, 1.0))

    return min_size + scaled * (max_size - min_size)


def add_3d_rotation_controls(fig, radius=1.9, z=1.05, n_frames=72):
    frames = []

    for i in range(n_frames):
        theta = 2 * np.pi * i / n_frames

        camera = dict(
            eye=dict(
                x=radius * np.cos(theta),
                y=radius * np.sin(theta),
                z=z
            )
        )

        frames.append(
            go.Frame(
                name=f"rotate_{i}",
                layout=dict(scene_camera=camera)
            )
        )

    fig.frames = frames

    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                x=0.02,
                y=0.02,
                xanchor="left",
                yanchor="bottom",
                showactive=False,
                buttons=[
                    dict(
                        label="▶ Rotate 360°",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=70, redraw=True),
                                transition=dict(duration=0),
                                fromcurrent=True,
                                mode="immediate"
                            )
                        ],
                    ),
                    dict(
                        label="⏸ Pause",
                        method="animate",
                        args=[
                            [None],
                            dict(
                                frame=dict(duration=0, redraw=False),
                                transition=dict(duration=0),
                                mode="immediate"
                            )
                        ],
                    ),
                ],
            )
        ]
    )

    return fig


# ============================================================
# 3. Part 1: Most-connected candidate extraction
# ============================================================

def rank_candidates_connected_to_query(query_genes, partial_corr_cutoff):
    edge_table = load_edges()
    node_final = load_node_final()
    ann = load_annotation()

    input_edges = edge_table[
        edge_table["source"].isin(query_genes) |
        edge_table["target"].isin(query_genes)
    ].copy()

    input_edges["input_gene"] = np.where(
        input_edges["source"].isin(query_genes),
        input_edges["source"],
        input_edges["target"]
    )

    input_edges["neighbor_gene"] = np.where(
        input_edges["source"].isin(query_genes),
        input_edges["target"],
        input_edges["source"]
    )

    input_edges = input_edges[input_edges["abs_partial_corr"] >= float(partial_corr_cutoff)].copy()

    if len(input_edges) == 0:
        raise ValueError(
            "No input-connected edges retained. Lower the partial correlation cutoff."
        )

    candidate_edges = input_edges[~input_edges["neighbor_gene"].isin(query_genes)].copy()

    if len(candidate_edges) == 0:
        raise ValueError(
            "No non-input candidate genes found. "
            "All retained neighbors are input genes, or the cutoff is too strict."
        )

    rank = (
        candidate_edges
        .groupby("neighbor_gene")
        .agg(
            n_input_connections=("input_gene", "size"),
            n_unique_input_genes=("input_gene", "nunique"),
            mean_abs_partial_corr=("abs_partial_corr", "mean"),
            max_abs_partial_corr=("abs_partial_corr", "max"),
            sum_abs_partial_corr=("abs_partial_corr", "sum"),
            mean_partial_corr=("partial_corr", "mean"),
            connected_input_genes=("input_gene", lambda x: ", ".join(pd.unique(x.astype(str))))
        )
        .reset_index()
    )

    rank = rank.sort_values(
        by=[
            "n_unique_input_genes",
            "sum_abs_partial_corr",
            "max_abs_partial_corr",
            "mean_abs_partial_corr"
        ],
        ascending=[False, False, False, False]
    )

    rank = rank.merge(
        node_final,
        left_on="neighbor_gene",
        right_on="Gene",
        how="left"
    )

    rank = rank.merge(
        ann,
        left_on="neighbor_gene",
        right_on="Gene",
        how="left",
        suffixes=("", "_ann")
    )

    if "GeneName" not in rank.columns:
        rank["GeneName"] = rank["neighbor_gene"]
    if "VC_ID" not in rank.columns:
        rank["VC_ID"] = rank["VC_ID_ann"] if "VC_ID_ann" in rank.columns else ""

    rank["GeneName"] = rank["GeneName"].fillna(rank["neighbor_gene"])
    rank["VC_ID"] = rank["VC_ID"].fillna("").astype(str).str.strip()

    return rank, input_edges


def build_candidate_subnetwork(
    query_genes,
    top_candidate_n,
    partial_corr_cutoff,
    max_edges_per_candidate,
):
    ann = load_annotation()
    node_final = load_node_final()

    candidate_rank, input_edges = rank_candidates_connected_to_query(
        query_genes=query_genes,
        partial_corr_cutoff=partial_corr_cutoff
    )

    top_candidates = candidate_rank.head(int(top_candidate_n)).copy()
    candidate_genes = top_candidates["neighbor_gene"].astype(str).tolist()

    edges_to_plot = input_edges[input_edges["neighbor_gene"].isin(candidate_genes)].copy()

    edges_to_plot = (
        edges_to_plot
        .sort_values(["neighbor_gene", "abs_partial_corr"], ascending=[True, False])
        .groupby("neighbor_gene")
        .head(int(max_edges_per_candidate))
        .drop_duplicates(subset=["source", "target"])
        .copy()
    )

    genes_to_plot = pd.unique(pd.concat([edges_to_plot["source"], edges_to_plot["target"]], axis=0))

    node_sub = pd.DataFrame({"Gene": genes_to_plot.astype(str)})
    node_sub = node_sub.merge(node_final, on="Gene", how="left")
    node_sub = node_sub.merge(ann, on="Gene", how="left", suffixes=("", "_ann"))

    if "GeneName" not in node_sub.columns:
        node_sub["GeneName"] = node_sub["Gene"]
    if "VC_ID" not in node_sub.columns:
        node_sub["VC_ID"] = node_sub["VC_ID_ann"] if "VC_ID_ann" in node_sub.columns else ""

    node_sub["GeneName"] = node_sub["GeneName"].fillna(node_sub["Gene"])
    node_sub["VC_ID"] = node_sub["VC_ID"].fillna("").astype(str).str.strip()
    node_sub["degree"] = pd.to_numeric(node_sub.get("degree", 1), errors="coerce").fillna(1)

    node_sub["node_type"] = np.where(
        node_sub["Gene"].isin(candidate_genes),
        "Top connected candidate",
        np.where(
            node_sub["Gene"].isin(query_genes),
            "Query gene",
            "Other"
        )
    )

    return candidate_rank, top_candidates, edges_to_plot, node_sub


def make_candidate_subnetwork_figure(
    query_genes,
    top_candidate_n,
    partial_corr_cutoff,
    max_edges_per_candidate,
    layout_seed,
    label_query_genes=True,
    label_top_candidates=True,
    query_color=DEFAULT_QUERY_COLOR,
):
    query_color = normalize_color(query_color, DEFAULT_QUERY_COLOR)

    candidate_rank, top_candidates, edges_to_plot, node_sub = build_candidate_subnetwork(
        query_genes=query_genes,
        top_candidate_n=top_candidate_n,
        partial_corr_cutoff=partial_corr_cutoff,
        max_edges_per_candidate=max_edges_per_candidate
    )

    G = nx.Graph()

    for _, row in node_sub.iterrows():
        G.add_node(
            row["Gene"],
            GeneName=row["GeneName"],
            VC_ID=row.get("VC_ID", ""),
            node_type=row["node_type"],
            degree=float(row["degree"])
        )

    for _, row in edges_to_plot.iterrows():
        source = str(row["source"])
        target = str(row["target"])

        if source in G.nodes and target in G.nodes:
            G.add_edge(
                source,
                target,
                partial_corr=float(row["partial_corr"]) if pd.notna(row["partial_corr"]) else 0.0,
                abs_partial_corr=float(row["abs_partial_corr"]) if pd.notna(row["abs_partial_corr"]) else 0.0,
                edge_type=str(row.get("edge_type", "unknown"))
            )

    pos = compute_layout_2d(G, seed=layout_seed)

    fig = go.Figure()

    edge_legend_seen = set()

    edge_type_color = {
        "positive": "rgba(170,35,35,0.92)",
        "negative": "rgba(35,85,165,0.92)",
        "unknown": "rgba(85,85,85,0.82)"
    }

    for u, v, d in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]

        edge_type = d.get("edge_type", "unknown")
        edge_color = edge_type_color.get(edge_type, "rgba(85,85,85,0.82)")

        partial_corr = float(d.get("partial_corr", 0.0))
        abs_partial_corr = float(d.get("abs_partial_corr", abs(partial_corr)))
        edge_width = edge_width_from_abs_corr(abs_partial_corr)

        show_legend = edge_type not in edge_legend_seen
        edge_legend_seen.add(edge_type)

        u_name = G.nodes[u].get("GeneName", u)
        v_name = G.nodes[v].get("GeneName", v)
        u_vc_id = G.nodes[u].get("VC_ID", "")
        v_vc_id = G.nodes[v].get("VC_ID", "")

        hover_text = (
            f"Source: {u_name} ({u})<br>"
            f"Source VC_ID: {u_vc_id}<br>"
            f"Target: {v_name} ({v})<br>"
            f"Target VC_ID: {v_vc_id}<br>"
            f"Partial correlation: {partial_corr:.4f}<br>"
            f"|Partial correlation|: {abs_partial_corr:.4f}<br>"
            f"Edge type: {edge_type}<br>"
            f"Edge width is scaled by |partial correlation|"
        )

        fig.add_trace(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines",
                line=dict(color=edge_color, width=edge_width),
                hovertext=hover_text,
                hoverinfo="text",
                name=f"{edge_type} edge",
                legendgroup=f"{edge_type} edge",
                showlegend=show_legend
            )
        )

    node_colors = {
        "Query gene": query_color,
        "Top connected candidate": "orange",
        "Other": "rgba(110,110,110,0.92)"
    }

    for node_type in ["Other", "Top connected candidate", "Query gene"]:
        nodes = [n for n in G.nodes if G.nodes[n]["node_type"] == node_type]

        if len(nodes) == 0:
            continue

        xs = [pos[n][0] for n in nodes]
        ys = [pos[n][1] for n in nodes]

        sizes = [
            node_size_from_degree(G.nodes[n].get("degree", 1))
            for n in nodes
        ]

        if node_type == "Query gene":
            sizes = [max(s, 16) for s in sizes]
        elif node_type == "Top connected candidate":
            sizes = [max(s, 14) for s in sizes]

        text = []
        hover = []

        for n in nodes:
            gname = G.nodes[n]["GeneName"]
            vc_id = G.nodes[n].get("VC_ID", "")
            displayed_degree = G.degree(n)
            full_degree = G.nodes[n].get("degree", "NA")

            hover.append(
                f"GeneName: {gname}<br>"
                f"Gene ID: {n}<br>"
                f"VC_ID: {vc_id}<br>"
                f"Node type: {node_type}<br>"
                f"Displayed degree: {displayed_degree}<br>"
                f"Full-network degree: {full_degree}<br>"
                f"Node size is scaled by full-network degree"
            )

            if node_type == "Query gene" and label_query_genes:
                text.append(f"{gname}")
            elif node_type == "Top connected candidate" and label_top_candidates:
                text.append(f"{gname}")
            else:
                text.append("")

        mode = "markers+text" if any(t != "" for t in text) else "markers"

        text_color = query_color if node_type == "Query gene" else "black"

        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode=mode,
                marker=dict(
                    size=sizes,
                    color=node_colors[node_type],
                    line=dict(color="black", width=1),
                    sizemode="diameter"
                ),
                text=text,
                textposition="top center",
                textfont=dict(size=12, color=text_color),
                hovertext=hover,
                hoverinfo="text",
                name=node_type,
                showlegend=True
            )
        )

    fig.update_layout(
        title=(
            f"Genes most connected to query genes in GGM network<br>"
            f"<sup>Top {top_candidate_n} non-input candidates | "
            f"|partial corr| ≥ {partial_corr_cutoff} | "
            f"max {max_edges_per_candidate} edges per candidate</sup>"
        ),
        template="plotly_white",
        height=760,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=20, r=40, t=120, b=40),
        legend=dict(x=1.02, y=1)
    )

    return fig, candidate_rank, top_candidates, edges_to_plot, node_sub


# ============================================================
# 4. Part 2: Full 3D network browser
# ============================================================

def make_full_3d_network_figure(
    query_genes,
    top_edge_number,
    layout_seed,
    show_query_labels,
    query_color,
    rotate_360=False,
    show_query_connected_edges=True,
    full_layout_method="fr",
):
    query_color = normalize_color(query_color, DEFAULT_QUERY_COLOR)

    edge_filt, node_filt = build_filtered_full_network(
        query_genes=query_genes,
        top_edge_number=top_edge_number,
        include_query_edges=show_query_connected_edges,
        query_edge_mode="query_only"
    )

    G = graph_from_edges_nodes(edge_filt, node_filt)

    if G.number_of_nodes() == 0:
        raise ValueError("No nodes available after filtering. Increase top edge number.")

    pos = compute_layout_3d(
        G,
        seed=layout_seed,
        layout_method=full_layout_method
    )

    fig = go.Figure()

    query_set = set(query_genes)

    # ------------------------------------------------------------
    # Full 3D non-query edges
    # ------------------------------------------------------------
    edge_type_color = {
        "positive": "rgba(150,60,60,0.62)",
        "negative": "rgba(60,95,160,0.62)",
        "unknown": "rgba(95,95,95,0.62)"
    }

    query_edge_type_color = {
        "positive": "rgba(170,35,35,0.95)",
        "negative": "rgba(35,85,165,0.95)",
        "unknown": "rgba(70,70,70,0.95)"
    }

    regular_edges = []
    query_edges = []

    for u, v, d in G.edges(data=True):
        edge_info = {
            "u": u,
            "v": v,
            "edge_type": d.get("edge_type", "unknown"),
            "partial_corr": float(d.get("partial_corr", 0.0)),
            "abs_partial_corr": float(d.get("abs_partial_corr", 0.0)),
            "is_query_edge": (u in query_set or v in query_set),
        }

        if edge_info["is_query_edge"]:
            query_edges.append(edge_info)
        else:
            regular_edges.append(edge_info)

    regular_edge_df = pd.DataFrame(regular_edges)

    if not regular_edge_df.empty:
        for edge_type, sub_edges in regular_edge_df.groupby("edge_type"):
            edge_x = []
            edge_y = []
            edge_z = []

            for _, row in sub_edges.iterrows():
                u = row["u"]
                v = row["v"]

                x0, y0, z0 = pos[u]
                x1, y1, z1 = pos[v]

                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
                edge_z.extend([z0, z1, None])

            fig.add_trace(
                go.Scatter3d(
                    x=edge_x,
                    y=edge_y,
                    z=edge_z,
                    mode="lines",
                    line=dict(
                        color=edge_type_color.get(edge_type, "rgba(95,95,95,0.62)"),
                        width=2.6
                    ),
                    hoverinfo="none",
                    name=f"{edge_type} background edges",
                    legendgroup=f"{edge_type} background edges",
                    showlegend=False
                )
            )

    # ------------------------------------------------------------
    # Query-connected edges: one trace per edge so width can scale
    # ------------------------------------------------------------
    query_edge_legend_seen = set()

    if show_query_connected_edges:
        for edge_info in query_edges:
            u = edge_info["u"]
            v = edge_info["v"]
            edge_type = edge_info["edge_type"]
            partial_corr = edge_info["partial_corr"]
            abs_partial_corr = edge_info["abs_partial_corr"]

            x0, y0, z0 = pos[u]
            x1, y1, z1 = pos[v]

            edge_width = query_edge_width_from_abs_corr(abs_partial_corr)
            edge_color = query_edge_type_color.get(edge_type, "rgba(70,70,70,0.95)")

            showlegend = edge_type not in query_edge_legend_seen
            query_edge_legend_seen.add(edge_type)

            u_name = G.nodes[u].get("GeneName", u)
            v_name = G.nodes[v].get("GeneName", v)
            u_vc_id = G.nodes[u].get("VC_ID", "")
            v_vc_id = G.nodes[v].get("VC_ID", "")

            hover_text = (
                f"Query-connected edge<br>"
                f"Source: {u_name} ({u})<br>"
                f"Source VC_ID: {u_vc_id}<br>"
                f"Target: {v_name} ({v})<br>"
                f"Target VC_ID: {v_vc_id}<br>"
                f"Partial correlation: {partial_corr:.4f}<br>"
                f"|Partial correlation|: {abs_partial_corr:.4f}<br>"
                f"Edge type: {edge_type}"
            )

            fig.add_trace(
                go.Scatter3d(
                    x=[x0, x1],
                    y=[y0, y1],
                    z=[z0, z1],
                    mode="lines",
                    line=dict(color=edge_color, width=edge_width),
                    hovertext=hover_text,
                    hoverinfo="text",
                    name=f"{edge_type} query-connected edge",
                    legendgroup=f"{edge_type} query-connected edge",
                    showlegend=showlegend
                )
            )

    # ------------------------------------------------------------
    # Full 3D nodes
    # ------------------------------------------------------------
    other_nodes = [n for n in G.nodes if not G.nodes[n].get("is_query", False)]
    query_nodes = [n for n in G.nodes if G.nodes[n].get("is_query", False)]

    def node_arrays(nodes):
        xs, ys, zs, hover, text, sizes = [], [], [], [], [], []

        for n in nodes:
            x, y, z = pos[n]
            xs.append(x)
            ys.append(y)
            zs.append(z)

            gname = G.nodes[n].get("GeneName", n)
            vc_id = G.nodes[n].get("VC_ID", "")
            displayed_degree = G.degree(n)
            full_degree = G.nodes[n].get("degree", "NA")

            hover.append(
                f"GeneName: {gname}<br>"
                f"Gene ID: {n}<br>"
                f"VC_ID: {vc_id}<br>"
                f"Displayed degree: {displayed_degree}<br>"
                f"Full-network degree: {full_degree}<br>"
                f"3D layout: {full_layout_method}"
            )

            text.append(f"{gname}")
            sizes.append(node_size_from_degree(full_degree, min_size=4.2, max_size=13.5))

        return xs, ys, zs, hover, text, sizes

    ox, oy, oz, ohover, otext, osizes = node_arrays(other_nodes)

    fig.add_trace(
        go.Scatter3d(
            x=ox,
            y=oy,
            z=oz,
            mode="markers",
            marker=dict(
                size=osizes,
                color="rgba(95,95,95,0.82)",
                sizemode="diameter"
            ),
            text=ohover,
            hoverinfo="text",
            name="Other genes",
            showlegend=True
        )
    )

    qx, qy, qz, qhover, qtext, qsizes = node_arrays(query_nodes)

    query_mode = "markers+text" if show_query_labels else "markers"

    query_sizes = [
        max(node_size_from_degree(G.nodes[n].get("degree", 1), min_size=18, max_size=36), 22)
        for n in query_nodes
    ]

    fig.add_trace(
        go.Scatter3d(
            x=qx,
            y=qy,
            z=qz,
            mode=query_mode,
            marker=dict(
                size=query_sizes,
                color=query_color,
                opacity=1,
                sizemode="diameter",
                line=dict(color="black", width=2.2)
            ),
            text=qtext if show_query_labels else qhover,
            hovertext=qhover,
            hoverinfo="text",
            textposition="top center",
            textfont=dict(size=16, color=query_color),
            name="Query genes",
            showlegend=True
        )
    )

    found_query = sorted(set(query_nodes))
    missing_from_displayed = sorted(set(query_genes) - set(found_query))

    layout_label = (
        "3D Fruchterman-Reingold"
        if full_layout_method == "fr"
        else "3D spherical layout"
    )

    title_extra = (
        f"Displayed nodes: {G.number_of_nodes()} | "
        f"Displayed edges: {G.number_of_edges()} | "
        f"Query genes shown: {len(found_query)}/{len(query_genes)} | "
        f"Layout: {layout_label}"
    )

    fig.update_layout(
        title=f"Full 3D GGM Network Browser<br><sup>{title_extra}</sup>",
        template="plotly_white",
        height=780,
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            camera=dict(eye=dict(x=1.9, y=1.9, z=1.05)),
            aspectmode="data"
        ),
        margin=dict(l=0, r=0, b=0, t=90),
        legend=dict(x=0.02, y=0.98)
    )

    if rotate_360:
        fig = add_3d_rotation_controls(fig)

    status = {
        "displayed_node_count": G.number_of_nodes(),
        "displayed_edge_count": G.number_of_edges(),
        "query_shown": found_query,
        "query_missing_from_displayed": missing_from_displayed,
        "rotate_360": rotate_360,
        "show_query_connected_edges": show_query_connected_edges,
        "full_layout_method": full_layout_method,
    }

    return fig, status


def make_query_summary(query_genes, missing_genes, query_status):
    preview = []

    for gid in query_genes[:15]:
        gname = gene_label(gid)
        if gname == gid:
            preview.append(gid)
        else:
            preview.append(f"{gname} ({gid})")

    preview_text = ", ".join(preview)

    if len(query_genes) > 15:
        preview_text += f", ... and {len(query_genes) - 15} more"

    children = [
        html.Div(html.Strong("Query summary")),
        html.Div(query_status),
        html.Div(f"Resolved query genes: {len(query_genes)}"),
        html.Div(preview_text)
    ]

    if missing_genes:
        missing_preview = ", ".join(missing_genes[:10])
        if len(missing_genes) > 10:
            missing_preview += f", ... and {len(missing_genes) - 10} more"

        children.append(
            html.Div(
                [
                    html.Strong("Not recognized: "),
                    missing_preview,
                    html.Br(),
                    "Please type a valid Gene ID, GeneName, or KEGG VC number from the annotation table, or upload a valid gene list."
                ],
                className="mt-2"
            )
        )

        return dbc.Alert(children, color="warning", className="mt-2")

    return dbc.Alert(children, color="info", className="mt-2")


# ============================================================
# 5. Layout
# ============================================================

available_gene_sets_text = ", ".join(GENE_SET_FILES.keys())

layout = dbc.Container(
    [
        html.H2("Network Browser", className="page-title"),

        html.P(
            "Explore the GGM-derived gene network. The first module extracts non-input genes "
            "that are most connected to the query gene set. The second module shows where the "
            "query genes are located in the full 3D network map.",
            className="lead"
        ),

        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Label("Query source"),
                        dcc.RadioItems(
                            id="network-query-source",
                            options=[
                                {"label": "Search gene / gene set", "value": "search"},
                                {"label": "Upload gene list", "value": "upload"},
                            ],
                            value="search",
                            inline=True,
                        ),
                    ],
                    md=12,
                ),
            ],
            className="mb-3",
        ),

        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Label("Search GeneName, Gene ID, KEGG VC number, or predefined gene set"),
                        dcc.Dropdown(
                            id="network-query-text",
                            options=load_gene_lookup_options(),
                            value=[f"__gene_set__::{DEFAULT_QUERY}"],
                            multi=True,
                            searchable=True,
                            clearable=True,
                            placeholder="Type GeneName, Gene ID, or KEGG VC number, e.g. tcpA, N900_RS01295, or VC_0828",
                        ),
                        html.Small(
                            f"Start typing to search GeneName/Gene ID/KEGG VC number. Available predefined gene sets: {available_gene_sets_text}",
                            className="text-muted"
                        ),
                        html.Div(id="network-query-summary"),
                    ],
                    md=5,
                ),
                dbc.Col(
                    [
                        html.Label("Upload gene list"),
                        dcc.Upload(
                            id="network-upload-gene-list",
                            children=html.Div(
                                [
                                    "Drag and drop or ",
                                    html.A("select a CSV/TSV/TXT gene list"),
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
                                "backgroundColor": "white",
                            },
                            multiple=False,
                        ),
                        html.Div(
                            id="network-upload-status",
                            className="text-muted mt-2"
                        ),
                    ],
                    md=4,
                ),
                dbc.Col(
                    [
                        html.Label("Query gene color"),
                        dbc.Input(
                            id="network-query-color",
                            type="text",
                            value=DEFAULT_QUERY_COLOR,
                            placeholder="e.g. red, black, #E7298A",
                        ),
                        html.Small(
                            "You can type a color name or hex code. Applied to query genes in both modules.",
                            className="text-muted"
                        ),
                    ],
                    md=3,
                ),
            ],
            className="mb-4",
        ),

        html.Hr(),

        # ======================================================
        # PART 1: Candidate extraction first
        # ======================================================

        html.H3("1. Extract genes most connected to the query gene list"),

        dbc.Alert(
            [
                html.Strong("How to use this module: "),
                html.Span(
                    "This module finds all GGM edges connected to the query genes, keeps edges above "
                    "the absolute partial-correlation cutoff, excludes the original query genes from "
                    "candidate ranking, and ranks non-input candidate genes by how strongly and broadly "
                    "they connect to the query set."
                ),
                html.Br(),
                html.Br(),
                html.Strong("Figure guide: "),
                html.Span("Query genes use your chosen color. "),
                html.Span("Orange nodes = top connected candidate genes. "),
                html.Span("Darker grey nodes = other genes. "),
                html.Span("Red edges = positive partial correlations; blue edges = negative partial correlations. "),
                html.Span("Thicker edges indicate larger |partial correlation|. "),
                html.Span("Larger node size indicates higher full-network degree. "),
                html.Br(),
                html.Br(),
                html.Strong("What does degree mean? "),
                html.Span(
                    "Degree is the number of network edges connected to a gene. "
                    "Displayed degree is the number of visible edges connected to a node in the plotted subnetwork. "
                    "Full-network degree is the number of edges connected to that gene in the complete GGM network. "
                    "A higher-degree gene is more connected and may act as a hub or coordinator in the inferred functional network."
                ),
                html.Br(),
                html.Span("Node labels show GeneName only. Hover over nodes and edges for detailed values."),
            ],
            color="secondary",
            className="mb-3"
        ),

        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Label("Top connected candidate genes"),
                        dbc.Input(
                            id="network-top-candidate-n",
                            type="number",
                            min=1,
                            max=100,
                            step=1,
                            value=10,
                        ),
                    ],
                    md=3,
                ),
                dbc.Col(
                    [
                        html.Label("|partial correlation| cutoff"),
                        dbc.Input(
                            id="network-partial-corr-cutoff",
                            type="number",
                            min=0,
                            max=1,
                            step=0.01,
                            value=0.05,
                        ),
                    ],
                    md=3,
                ),
                dbc.Col(
                    [
                        html.Label("Max edges per candidate"),
                        dbc.Input(
                            id="network-max-edges-per-candidate",
                            type="number",
                            min=1,
                            max=100,
                            step=1,
                            value=10,
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
                        html.Label("Subnetwork display options"),
                        dcc.Checklist(
                            id="network-candidate-display-options",
                            options=[
                                {"label": "Label query genes", "value": "label_query"},
                                {"label": "Label top candidates", "value": "label_candidates"},
                            ],
                            value=["label_query", "label_candidates"],
                            inline=True,
                        ),
                    ],
                    md=8,
                ),
            ],
            className="mb-3",
        ),

        dcc.Tabs(
            id="network-candidate-tabs",
            value="plot",
            children=[
                dcc.Tab(label="Subnetwork Plot", value="plot"),
                dcc.Tab(label="Candidate Ranking Table", value="ranking"),
                dcc.Tab(label="Top Candidate Table", value="top"),
                dcc.Tab(label="Edges Table", value="edges"),
                dcc.Tab(label="Nodes Table", value="nodes"),
            ],
        ),

        html.Br(),

        dcc.Loading(
            type="circle",
            children=[
                html.Div(id="network-candidate-content")
            ],
        ),

        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Button(
                            "Download candidate ranking",
                            id="network-download-candidate-ranking-button",
                            color="secondary",
                            size="sm",
                            className="mt-3"
                        ),
                        dcc.Download(id="network-download-candidate-ranking"),
                    ],
                    md=3,
                ),
            ],
            className="mb-5",
        ),

        html.Hr(),

        # ======================================================
        # PART 2: Full 3D map second
        # ======================================================

        html.H3("2. Query genes in the full 3D network map"),

        dbc.Alert(
            [
                html.Strong("Optional slow module: "),
                html.Span(
                    "The full 3D network map can take time to compute and render, especially when many edges are shown. "
                    "It is turned off by default. Enable it only when you want to inspect query gene positions in the full 3D map."
                ),
            ],
            color="warning",
            className="mb-2"
        ),

        dcc.Checklist(
            id="network-show-full-map",
            options=[
                {
                    "label": "Show Module 2 full 3D network map. This may take time to compute and render.",
                    "value": "show_full_map",
                }
            ],
            value=[],
            className="mb-3",
        ),

        html.Div(
            id="network-full-map-section",
            style={"display": "none"},
            children=[
                dbc.Alert(
                    [
                        html.Strong("How to use this module: "),
                        html.Span(
                            "This module shows where the query genes are located in the full 3D GGM network. "
                            "The query genes are highlighted with larger dots in the color you choose. "
                            "You can optionally show thick query-connected edges, which makes it easier to see "
                            "the direct network neighborhood of your gene list."
                        ),
                        html.Br(),
                        html.Br(),
                        html.Strong("3D layout options: "),
                        html.Span(
                            "The Fruchterman-Reingold option uses a force-directed layout similar to R/igraph "
                            "layout_with_fr(..., dim = 3). The spherical layout distributes nodes on a sphere-like "
                            "3D map, with high-degree nodes placed slightly more centrally."
                        ),
                        html.Br(),
                        html.Br(),
                        html.Strong("Rotation option: "),
                        html.Span(
                            "Enable 360-degree rotation to add Play/Pause controls to the 3D figure. "
                            "This is useful for screen recording or exporting a rotating network view."
                        ),
                    ],
                    color="secondary",
                    className="mb-3"
                ),

                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Label("Top edges shown in full network"),
                                dbc.Input(
                                    id="network-top-edge-number",
                                    type="number",
                                    min=100,
                                    max=20000,
                                    step=100,
                                    value=3000,
                                ),
                            ],
                            md=3,
                        ),
                        dbc.Col(
                            [
                                html.Label("3D layout seed"),
                                dbc.Input(
                                    id="network-layout-seed",
                                    type="number",
                                    value=222,
                                    step=1,
                                ),
                            ],
                            md=2,
                        ),
                        dbc.Col(
                            [
                                html.Label("Full 3D layout algorithm"),
                                dcc.Dropdown(
                                    id="network-full-layout-method",
                                    options=[
                                        {
                                            "label": "3D Fruchterman-Reingold force-directed layout",
                                            "value": "fr",
                                        },
                                        {
                                            "label": "3D spherical layout",
                                            "value": "sphere",
                                        },
                                    ],
                                    value="fr",
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
                                html.Label("Full-map display options"),
                                dcc.Checklist(
                                    id="network-full-display-options",
                                    options=[
                                        {"label": "Show query gene labels", "value": "show_query_labels"},
                                        {"label": "Show connected edges with query gene list", "value": "show_query_connected_edges"},
                                        {"label": "Enable 360-degree rotation controls", "value": "rotate_360"},
                                    ],
                                    value=["show_query_connected_edges"],
                                    inline=False,
                                ),
                            ],
                            md=8,
                        ),
                    ],
                    className="mb-3",
                ),

                dcc.Loading(
                    type="circle",
                    children=[
                        dcc.Graph(id="network-full-3d-plot", style={"width": "100%"})
                    ],
                ),

                html.Div(id="network-full-status"),
            ],
        ),
    ],
    fluid=True
)


# ============================================================
# 6. Callbacks
# ============================================================

@dash.callback(
    Output("network-candidate-content", "children"),
    Input("network-candidate-tabs", "value"),
    Input("network-query-source", "value"),
    Input("network-query-text", "value"),
    Input("network-upload-gene-list", "contents"),
    State("network-upload-gene-list", "filename"),
    Input("network-query-color", "value"),
    Input("network-top-candidate-n", "value"),
    Input("network-partial-corr-cutoff", "value"),
    Input("network-max-edges-per-candidate", "value"),
    Input("network-layout-seed", "value"),
    Input("network-candidate-display-options", "value"),
)
def update_candidate_network(
    active_tab,
    query_source,
    query_text,
    upload_contents,
    upload_filename,
    query_color,
    top_candidate_n,
    partial_corr_cutoff,
    max_edges_per_candidate,
    layout_seed,
    display_options,
):
    try:
        display_options = display_options or []

        query_genes, missing_genes, query_status = get_query_genes(
            query_source=query_source,
            query_text=query_text,
            upload_contents=upload_contents,
            upload_filename=upload_filename
        )

        label_query = "label_query" in display_options
        label_candidates = "label_candidates" in display_options

        fig, candidate_rank, top_candidates, edges_to_plot, node_sub = make_candidate_subnetwork_figure(
            query_genes=query_genes,
            top_candidate_n=int(top_candidate_n),
            partial_corr_cutoff=float(partial_corr_cutoff),
            max_edges_per_candidate=int(max_edges_per_candidate),
            layout_seed=int(layout_seed),
            label_query_genes=label_query,
            label_top_candidates=label_candidates,
            query_color=query_color,
        )

        if active_tab == "plot":
            return dcc.Graph(figure=fig, style={"width": "100%"})

        if active_tab == "ranking":
            df = candidate_rank.copy()
        elif active_tab == "top":
            df = top_candidates.copy()
        elif active_tab == "edges":
            df = edges_to_plot.copy()
        else:
            df = node_sub.copy()

        for col in df.select_dtypes(include=[np.number]).columns:
            df[col] = df[col].round(4)

        return dash_table.DataTable(
            data=df.to_dict("records"),
            columns=[{"name": c, "id": c} for c in df.columns],
            page_size=15,
            filter_action="native",
            sort_action="native",
            style_table={"overflowX": "auto"},
            style_cell={
                "textAlign": "left",
                "fontFamily": "Arial",
                "fontSize": "13px",
                "padding": "6px",
                "minWidth": "120px",
                "whiteSpace": "normal",
            },
            style_header={
                "fontWeight": "bold",
                "backgroundColor": "#f1f3f5",
            },
        )

    except Exception as e:
        return dbc.Alert(str(e), color="danger")


@dash.callback(
    Output("network-full-map-section", "style"),
    Input("network-show-full-map", "value"),
)
def toggle_full_map_section(show_full_map_values):
    """Show/hide Module 2 immediately when the checkbox is toggled."""
    if "show_full_map" in (show_full_map_values or []):
        return {"display": "block"}
    return {"display": "none"}


@dash.callback(
    Output("network-full-3d-plot", "figure"),
    Output("network-full-status", "children"),
    Output("network-query-summary", "children"),
    Output("network-upload-status", "children"),
    Input("network-show-full-map", "value"),
    Input("network-query-source", "value"),
    Input("network-query-text", "value"),
    Input("network-upload-gene-list", "contents"),
    State("network-upload-gene-list", "filename"),
    Input("network-query-color", "value"),
    Input("network-top-edge-number", "value"),
    Input("network-layout-seed", "value"),
    Input("network-full-layout-method", "value"),
    Input("network-full-display-options", "value"),
)
def update_full_network(
    show_full_map_values,
    query_source,
    query_text,
    upload_contents,
    upload_filename,
    query_color,
    top_edge_number,
    layout_seed,
    full_layout_method,
    display_options,
):
    try:
        display_options = display_options or []

        query_genes, missing_genes, query_status = get_query_genes(
            query_source=query_source,
            query_text=query_text,
            upload_contents=upload_contents,
            upload_filename=upload_filename
        )

        query_summary = make_query_summary(
            query_genes=query_genes,
            missing_genes=missing_genes,
            query_status=query_status
        )

        upload_status = query_status if query_source == "upload" else ""

        if "show_full_map" not in (show_full_map_values or []):
            fig = go.Figure()
            fig.update_layout(
                template="plotly_white",
                title="Full 3D network map is disabled",
                annotations=[
                    dict(
                        text="Enable the Module 2 checkbox to compute and display the full 3D map.",
                        x=0.5,
                        y=0.5,
                        showarrow=False,
                    )
                ],
                height=420,
            )
            status_children = dbc.Alert(
                "Module 2 is currently disabled. Turn on the checkbox above to compute the full 3D map; this may take time.",
                color="warning",
                className="mt-2",
            )
            return fig, status_children, query_summary, upload_status

        show_query_labels = "show_query_labels" in display_options
        rotate_360 = "rotate_360" in display_options
        show_query_connected_edges = "show_query_connected_edges" in display_options

        fig, status = make_full_3d_network_figure(
            query_genes=query_genes,
            top_edge_number=int(top_edge_number),
            layout_seed=int(layout_seed),
            show_query_labels=show_query_labels,
            query_color=query_color,
            rotate_360=rotate_360,
            show_query_connected_edges=show_query_connected_edges,
            full_layout_method=full_layout_method,
        )

        layout_text = (
            "3D spherical layout is used."
            if full_layout_method == "sphere"
            else "3D Fruchterman-Reingold force-directed layout is used."
        )

        edge_text = (
            "Query-connected edges are shown."
            if show_query_connected_edges
            else "Query-connected edges are not forced into the display."
        )

        rotation_text = (
            "360-degree rotation controls are enabled."
            if rotate_360
            else "360-degree rotation controls are disabled."
        )

        status_children = [
            dbc.Alert(
                [
                    html.Strong("Full-network display summary: "),
                    f"{status['displayed_node_count']} nodes and "
                    f"{status['displayed_edge_count']} edges are displayed. "
                    f"{len(status['query_shown'])}/{len(query_genes)} query genes are present in the displayed map. ",
                    html.Br(),
                    layout_text,
                    " ",
                    edge_text,
                    " ",
                    rotation_text,
                ],
                color="info",
                className="mt-3"
            )
        ]

        if status["query_missing_from_displayed"]:
            missing_labels = [
                f"{gene_label(g)} ({g})"
                for g in status["query_missing_from_displayed"][:10]
            ]

            status_children.append(
                dbc.Alert(
                    [
                        html.Strong("Some query genes were not shown in the filtered 3D map: "),
                        ", ".join(missing_labels),
                        html.Br(),
                        "Try increasing the number of top edges shown or enabling query-connected edges."
                    ],
                    color="warning"
                )
            )

        return fig, html.Div(status_children), query_summary, upload_status

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

        alert = dbc.Alert(str(e), color="danger", className="mt-2")

        return fig, alert, alert, ""


@dash.callback(
    Output("network-download-candidate-ranking", "data"),
    Input("network-download-candidate-ranking-button", "n_clicks"),
    State("network-query-source", "value"),
    State("network-query-text", "value"),
    State("network-upload-gene-list", "contents"),
    State("network-upload-gene-list", "filename"),
    State("network-partial-corr-cutoff", "value"),
    prevent_initial_call=True,
)
def download_candidate_ranking(
    n_clicks,
    query_source,
    query_text,
    upload_contents,
    upload_filename,
    partial_corr_cutoff,
):
    if not n_clicks:
        return no_update

    query_genes, missing_genes, query_status = get_query_genes(
        query_source=query_source,
        query_text=query_text,
        upload_contents=upload_contents,
        upload_filename=upload_filename
    )

    candidate_rank, input_edges = rank_candidates_connected_to_query(
        query_genes=query_genes,
        partial_corr_cutoff=float(partial_corr_cutoff)
    )

    filename = "network_browser_candidate_ranking.csv"

    return dcc.send_data_frame(candidate_rank.to_csv, filename, index=False)
