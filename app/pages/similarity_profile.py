import os
import math
import base64
import io
from functools import lru_cache

import numpy as np
import pandas as pd
import dash
from dash import html, dcc, dash_table, Input, Output, State, no_update
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.nonparametric.smoothers_lowess import lowess


dash.register_page(__name__, path="/similarity-profile", name="Similarity Profile")


# ============================================================
# 0. Paths and constants
# ============================================================

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(APP_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

RAW_DATA_FILE = os.path.join(
    DATA_DIR,
    "raw",
    "Spatial_temporal_MultiSCAST_FC_final_capping.csv"
)

ANNOTATION_FILE = os.path.join(
    DATA_DIR,
    "annotation",
    "new_annotations_with_uniprot_names.csv"
)

GENE_SET_FILES = {
    "motV": os.path.join(DATA_DIR, "gene_sets", "motV.csv"),
    # Add more predefined gene sets here, for example:
    # "Flagella": os.path.join(DATA_DIR, "gene_sets", "Flagella_genes.csv"),
    # "Tcp": os.path.join(DATA_DIR, "gene_sets", "Tcp_genes.csv"),
    # "Bile resistance": os.path.join(DATA_DIR, "gene_sets", "bile_resistance_genes.csv"),
}

SPACEPOINTS = [
    "st", "SI1", "SI2", "SI3", "SI4", "SI5",
    "SI6", "SI7", "SI8", "SI9", "ce", "co"
]

FULL_TIMEPOINTS = ["1h", "3h", "6h", "12h", "24h"]


# ============================================================
# 1. Data loading
# ============================================================

@lru_cache(maxsize=1)
def load_raw_data():
    """
    Load original spatial-temporal fitness table.

    Required columns:
      Gene, Time, Space, logFC
    """
    if not os.path.exists(RAW_DATA_FILE):
        raise FileNotFoundError(f"Raw data file not found: {RAW_DATA_FILE}")

    df = pd.read_csv(RAW_DATA_FILE)

    required = {"Gene", "Time", "Space", "logFC"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in raw data: {missing}")

    df = df[["Gene", "Time", "Space", "logFC"]].copy()
    df["Gene"] = df["Gene"].astype(str)
    df["Time"] = df["Time"].astype(str)
    df["Space"] = df["Space"].astype(str)
    df["logFC"] = pd.to_numeric(df["logFC"], errors="coerce")

    return df


@lru_cache(maxsize=1)
def load_annotation():
    """
    Load gene annotation table.

    Expected columns:
      locus_ID, gene_name
    """
    if not os.path.exists(ANNOTATION_FILE):
        return pd.DataFrame(columns=["Gene", "GeneName"])

    ann = pd.read_csv(ANNOTATION_FILE)

    if "locus_ID" not in ann.columns:
        return pd.DataFrame(columns=["Gene", "GeneName"])

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
def load_gene_lookup_options():
    """
    Build searchable dropdown options for manual query.

    Users can search by either GeneName or Gene ID.  The value passed
    downstream is always the canonical locus ID, so all existing plotting
    code continues to work.  Predefined gene sets are also included as
    selectable options.
    """
    raw_df = load_raw_data()
    ann = load_annotation()

    lookup = pd.DataFrame({"Gene": sorted(raw_df["Gene"].astype(str).unique())})
    lookup = lookup.merge(ann, on="Gene", how="left")
    lookup["GeneName"] = lookup["GeneName"].fillna(lookup["Gene"])
    lookup["DisplayLabel"] = lookup["GeneName"] + " | " + lookup["Gene"]

    gene_options = [
        {"label": row["DisplayLabel"], "value": row["Gene"]}
        for _, row in lookup.sort_values(["GeneName", "Gene"]).iterrows()
    ]

    gene_set_options = [
        {"label": f"Gene set: {key}", "value": f"__gene_set__::{key}"}
        for key in sorted(GENE_SET_FILES.keys(), key=lambda x: x.lower())
    ]

    return gene_set_options + gene_options


@lru_cache(maxsize=32)
def load_gene_set(gene_set_name):
    """
    Load predefined query gene set.

    Expected columns:
      locus_ID

    Optional columns:
      gene, category, Color
    """
    path = GENE_SET_FILES.get(gene_set_name)

    if path is None or not os.path.exists(path):
        raise FileNotFoundError(f"Gene set file not found for: {gene_set_name}")

    gs = pd.read_csv(path)

    if "locus_ID" not in gs.columns:
        raise ValueError("Gene set file must contain column: locus_ID")

    if "gene" not in gs.columns:
        gs["gene"] = gs["locus_ID"]

    if "category" not in gs.columns:
        gs["category"] = gene_set_name

    if "Color" not in gs.columns:
        gs["Color"] = "#2ca02c"

    gs = gs.copy()
    gs["locus_ID"] = gs["locus_ID"].astype(str)
    gs["gene"] = gs["gene"].fillna("").astype(str)

    gs["Gene_label"] = np.where(
        gs["gene"].str.strip() != "",
        gs["gene"],
        gs["locus_ID"]
    )

    return gs[["locus_ID", "gene", "category", "Color", "Gene_label"]].drop_duplicates()


# ============================================================
# 2. Query / upload helpers
# ============================================================

def parse_gene_tokens(text):
    """
    Parse manual gene input.

    Accepts:
      - a string with comma/newline/tab/semicolon/space-separated genes
      - a list from a searchable Dash Dropdown with multi=True
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
        "locus_id", "locus_id,", "genename", "gene_name"
    }

    tokens = [x for x in tokens if x.lower() not in bad_headers]

    return list(dict.fromkeys(tokens))


def get_predefined_gene_set_key(query_text):
    """
    If query_text exactly matches one predefined gene set name, return the canonical key.
    Matching is case-insensitive. Also supports encoded dropdown values.
    """
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


def resolve_gene_ids(gene_tokens):
    """
    Convert gene IDs or gene names into locus IDs used in the raw data.
    """
    raw_df = load_raw_data()
    ann = load_annotation()

    valid_gene_ids = set(raw_df["Gene"].astype(str).unique())

    gene_name_to_ids = {}

    for _, row in ann.iterrows():
        gid = str(row["Gene"])
        gname = str(row["GeneName"])

        gene_name_to_ids.setdefault(gid.lower(), []).append(gid)
        gene_name_to_ids.setdefault(gname.lower(), []).append(gid)

    resolved = []
    missing = []

    for token in gene_tokens:
        token_str = str(token).strip()
        token_lower = token_str.lower()

        if token_str in valid_gene_ids:
            resolved.append(token_str)
        elif token_lower in gene_name_to_ids:
            resolved.extend(gene_name_to_ids[token_lower])
        else:
            missing.append(token_str)

    resolved = list(dict.fromkeys(resolved))

    return resolved, missing


def parse_uploaded_gene_file(contents, filename):
    """
    Parse uploaded gene list.

    Supports:
      csv, tsv, txt

    Preferred columns:
      locus_ID, Gene, gene, GeneName, gene_name

    If no known column is found, uses the first column.
    """
    if contents is None:
        return []

    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)

    filename = filename or ""

    text = decoded.decode("utf-8", errors="replace")

    if filename.endswith(".csv"):
        df = pd.read_csv(io.StringIO(text))
    elif filename.endswith(".tsv"):
        df = pd.read_csv(io.StringIO(text), sep="\t")
    elif filename.endswith(".txt"):
        return parse_gene_tokens(text)
    else:
        return parse_gene_tokens(text)

    candidate_cols = [
        "locus_ID",
        "locus_id",
        "Gene",
        "gene",
        "GeneName",
        "gene_name"
    ]

    selected_col = None
    for col in candidate_cols:
        if col in df.columns:
            selected_col = col
            break

    if selected_col is None:
        selected_col = df.columns[0]

    genes = (
        df[selected_col]
        .dropna()
        .astype(str)
        .str.strip()
        .tolist()
    )

    genes = [g for g in genes if g != ""]

    return list(dict.fromkeys(genes))


def make_custom_gene_set_df(gene_ids, label="Custom gene set"):
    """
    Create the same structure as predefined gene set files.
    """
    ann = load_annotation()
    gene_label_map = dict(zip(ann["Gene"], ann["GeneName"]))

    rows = []

    for gid in gene_ids:
        rows.append(
            {
                "locus_ID": gid,
                "gene": gene_label_map.get(gid, gid),
                "category": label,
                "Color": "#2ca02c",
                "Gene_label": gene_label_map.get(gid, gid),
            }
        )

    return pd.DataFrame(rows)



def expand_predefined_gene_set_tokens(tokens):
    """
    Allow the searchable dropdown/manual input to mix individual genes and
    predefined gene sets. Encoded values use __gene_set__::<name>.
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
            gs = load_gene_set(key)
            expanded_tokens.extend(gs["locus_ID"].astype(str).tolist())
        else:
            expanded_tokens.append(token_str)

    return list(dict.fromkeys(expanded_tokens))

def get_query_gene_set(query_source, query_text, upload_contents, upload_filename):
    """
    Return query gene set dataframe with columns:
      locus_ID, gene, category, Color, Gene_label

    query_source:
      - "search": use query_text. If it matches a predefined gene set, use that gene set.
                  Otherwise resolve gene IDs / GeneNames from annotation.
      - "upload": use uploaded file.
    """
    if query_source == "search":
        query_text = query_text or ""

        predefined_key = get_predefined_gene_set_key(query_text)
        if predefined_key is not None:
            return load_gene_set(predefined_key), [], f"Using predefined gene set: {predefined_key}"

        tokens = parse_gene_tokens(query_text)
        tokens = expand_predefined_gene_set_tokens(tokens)
        resolved, missing = resolve_gene_ids(tokens)

        if len(resolved) == 0:
            available_sets = ", ".join(GENE_SET_FILES.keys())
            raise ValueError(
                "No valid genes found from the query. "
                "Please enter a valid gene ID, gene name, or predefined gene set name. "
                f"Available predefined gene sets: {available_sets}"
            )

        return make_custom_gene_set_df(resolved, label="Searched query"), missing, "Using searched query genes."

    if query_source == "upload":
        tokens = parse_uploaded_gene_file(upload_contents, upload_filename)
        resolved, missing = resolve_gene_ids(tokens)

        if len(resolved) == 0:
            raise ValueError(
                "No valid genes found from uploaded file. "
                "Please upload a file containing locus_ID, Gene, gene, GeneName, or gene_name."
            )

        return make_custom_gene_set_df(resolved, label="Uploaded query"), missing, f"Using uploaded file: {upload_filename}"

    raise ValueError(f"Unknown query source: {query_source}")


def build_query_summary(query_gene_set_df, missing_genes, query_status, single_query_mode):
    """
    Build a user-facing summary below the search box.
    """
    if query_gene_set_df is None or query_gene_set_df.empty:
        return dbc.Alert(
            "No query genes were resolved. Please type a gene ID, a gene name, or a predefined gene set.",
            color="warning",
            className="mt-2"
        )

    summary_df = query_gene_set_df[["locus_ID", "Gene_label"]].drop_duplicates().copy()
    n_genes = summary_df.shape[0]

    preview_items = []
    max_preview = 12

    for _, row in summary_df.head(max_preview).iterrows():
        gene_label = str(row["Gene_label"])
        gene_id = str(row["locus_ID"])

        if gene_label == gene_id:
            preview_items.append(gene_id)
        else:
            preview_items.append(f"{gene_label} ({gene_id})")

    preview_text = ", ".join(preview_items)

    if n_genes > max_preview:
        preview_text += f", ... and {n_genes - max_preview} more"

    if single_query_mode:
        mode_text = "Single-gene query: this gene is directly used as the query pattern."
    else:
        mode_text = f"Multi-gene query: {n_genes} genes are used to calculate the median query pattern."

    children = [
        html.Div(html.Strong("Query summary")),
        html.Div(query_status),
        html.Div(mode_text),
        html.Div(f"Resolved genes: {preview_text}")
    ]

    if missing_genes:
        missing_preview = ", ".join(missing_genes[:12])
        if len(missing_genes) > 12:
            missing_preview += f", ... and {len(missing_genes) - 12} more"

        children.append(
            html.Div(
                [
                    html.Strong("Not found in annotation/raw data: "),
                    missing_preview,
                    html.Br(),
                    "Please type a valid gene ID or try another gene name."
                ],
                className="mt-2"
            )
        )

        return dbc.Alert(children, color="warning", className="mt-2")

    return dbc.Alert(children, color="info", className="mt-2")


# ============================================================
# 3. Similarity calculation
# ============================================================

def get_time_window(start_time, end_time):
    start_idx = FULL_TIMEPOINTS.index(start_time)
    end_idx = FULL_TIMEPOINTS.index(end_time)

    if start_idx > end_idx:
        start_idx, end_idx = end_idx, start_idx

    return FULL_TIMEPOINTS[start_idx:(end_idx + 1)]


def get_space_window(start_space, end_space):
    start_idx = SPACEPOINTS.index(start_space)
    end_idx = SPACEPOINTS.index(end_space)

    if start_idx > end_idx:
        start_idx, end_idx = end_idx, start_idx

    return SPACEPOINTS[start_idx:(end_idx + 1)]


def build_gene_matrix_table(df, timepoints, spacepoints):
    """
    Convert long table into a Gene x (Time, Space) matrix.

    Missing values are filled with 0.
    """
    df2 = df[df["Time"].isin(timepoints)].copy()
    df2 = df2[df2["Space"].isin(spacepoints)].copy()

    all_cols = pd.MultiIndex.from_product(
        [timepoints, spacepoints],
        names=["Time", "Space"]
    )

    wide = (
        df2
        .pivot_table(
            index="Gene",
            columns=["Time", "Space"],
            values="logFC",
            aggfunc="mean"
        )
        .reindex(columns=all_cols)
        .fillna(0)
    )

    return wide


def matrix_from_row(row_values, n_time, n_space):
    return np.asarray(row_values, dtype=float).reshape(n_time, n_space)


def multivariate_dtw_distance(mat_a, mat_b):
    """
    Multivariate DTW over time.

    Each timepoint is a vector across intestinal spacepoints.
    Local cost = Euclidean distance between two spatial vectors.
    """
    n, m = mat_a.shape[0], mat_b.shape[0]

    dp = np.full((n + 1, m + 1), np.inf)
    dp[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = np.linalg.norm(mat_a[i - 1, :] - mat_b[j - 1, :])
            dp[i, j] = cost + min(
                dp[i - 1, j],
                dp[i, j - 1],
                dp[i - 1, j - 1]
            )

    return float(dp[n, m])


def cosine_distance(vec_a, vec_b):
    denom = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)

    if denom == 0:
        return np.inf

    sim = np.dot(vec_a, vec_b) / denom

    return float(1 - sim)


def compute_similarity_from_gene_set_df(
    query_gene_set_df,
    start_time,
    end_time,
    start_space,
    end_space
):
    """
    Compute DTW and cosine ranking for all genes using a query gene set dataframe.
    If the query contains only 1 gene, directly use that gene as the query pattern.
    If the query contains 2+ genes, use the median pattern.
    """
    timepoints = get_time_window(start_time, end_time)
    spacepoints = get_space_window(start_space, end_space)

    df = load_raw_data()
    ann = load_annotation()

    input_genes = (
        query_gene_set_df["locus_ID"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    wide = build_gene_matrix_table(df, tuple(timepoints), tuple(spacepoints))

    existing_input_genes = [g for g in input_genes if g in wide.index]

    if len(existing_input_genes) == 0:
        raise ValueError(
            "None of the query genes were found in the raw fitness data."
        )

    input_matrix = wide.loc[existing_input_genes].values

    if len(existing_input_genes) == 1:
        query_vec = input_matrix[0]
    else:
        query_vec = np.median(input_matrix, axis=0)

    n_time = len(timepoints)
    n_space = len(spacepoints)
    query_mat = matrix_from_row(query_vec, n_time, n_space)

    results = []

    for gene, row in wide.iterrows():
        vec = row.values.astype(float)
        mat = matrix_from_row(vec, n_time, n_space)

        dtw_dist = multivariate_dtw_distance(mat, query_mat)
        cos_dist = cosine_distance(vec, query_vec)

        results.append(
            {
                "Gene": gene,
                "DTW_dist": dtw_dist,
                "Cosine_dist": cos_dist,
            }
        )

    sim_df = pd.DataFrame(results)
    sim_df = sim_df.merge(ann, on="Gene", how="left")
    sim_df["GeneName"] = sim_df["GeneName"].fillna(sim_df["Gene"])
    sim_df = sim_df.sort_values("DTW_dist").reset_index(drop=True)

    return sim_df


def make_top_compare(sim_df, top_n):
    top_n = int(top_n)
    top_n = max(1, min(top_n, len(sim_df)))

    top_dtw = (
        sim_df
        .sort_values("DTW_dist")
        .head(top_n)
        .copy()
    )
    top_dtw["rank_DTW"] = np.arange(1, len(top_dtw) + 1)

    top_cosine = (
        sim_df
        .sort_values("Cosine_dist")
        .head(top_n)
        .copy()
    )
    top_cosine["rank_cosine"] = np.arange(1, len(top_cosine) + 1)

    top_dtw = top_dtw[["Gene", "GeneName", "DTW_dist", "rank_DTW"]]
    top_cosine = top_cosine[["Gene", "GeneName", "Cosine_dist", "rank_cosine"]]

    merged = pd.merge(
        top_dtw,
        top_cosine,
        on=["Gene", "GeneName"],
        how="outer"
    )

    merged["in_DTW_top_n"] = merged["rank_DTW"].notna()
    merged["in_cosine_top_n"] = merged["rank_cosine"].notna()
    merged["in_both"] = merged["in_DTW_top_n"] & merged["in_cosine_top_n"]

    merged = merged.sort_values(
        by=["in_both", "rank_DTW", "rank_cosine"],
        ascending=[False, True, True],
        na_position="last"
    )

    return merged.reset_index(drop=True)


# ============================================================
# 4. Plot helpers
# ============================================================

def fit_curve(x, y, method="poly2", n_points=100):
    """
    Fit curve using raw, polynomial, or LOESS smoothing.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) == 0:
        return np.array([]), np.array([])

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    if method == "raw":
        return x, y

    unique_x = np.unique(x)

    if len(unique_x) < 2:
        return x, y

    xfit = np.linspace(np.min(x), np.max(x), n_points)

    if method == "loess":
        frac = min(1.0, max(0.45, 4 / max(len(x), 1)))

        smoothed = lowess(
            endog=y,
            exog=x,
            frac=frac,
            return_sorted=True
        )

        xs = smoothed[:, 0]
        ys = smoothed[:, 1]

        unique_df = (
            pd.DataFrame({"x": xs, "y": ys})
            .groupby("x", as_index=False)["y"]
            .mean()
        )

        if unique_df.shape[0] < 2:
            return x, y

        yfit = np.interp(xfit, unique_df["x"], unique_df["y"])

        return xfit, yfit

    if method == "poly3":
        degree = 3
    else:
        degree = 2

    degree = min(degree, len(unique_x) - 1)

    coef = np.polyfit(x, y, degree)
    yfit = np.polyval(coef, xfit)

    return xfit, yfit


def get_top_genes(top_compare, method):
    if method == "DTW":
        genes = (
            top_compare[top_compare["in_DTW_top_n"]]
            .sort_values("rank_DTW")["Gene"]
            .tolist()
        )
    elif method == "cosine":
        genes = (
            top_compare[top_compare["in_cosine_top_n"]]
            .sort_values("rank_cosine")["Gene"]
            .tolist()
        )
    elif method == "both":
        genes = (
            top_compare[top_compare["in_both"]]
            .sort_values(["rank_DTW", "rank_cosine"])["Gene"]
            .tolist()
        )
    else:
        genes = top_compare["Gene"].tolist()

    return list(dict.fromkeys(genes))


def prepare_plot_data(
    query_source,
    query_text,
    upload_contents,
    upload_filename,
    start_time,
    end_time,
    start_space,
    end_space,
    top_n,
    top_method
):
    timepoints = get_time_window(start_time, end_time)
    spacepoints = get_space_window(start_space, end_space)

    df = load_raw_data()
    ann = load_annotation()

    query_gene_set_df, missing_genes, query_status = get_query_gene_set(
        query_source=query_source,
        query_text=query_text,
        upload_contents=upload_contents,
        upload_filename=upload_filename
    )

    input_genes = (
        query_gene_set_df["locus_ID"]
        .astype(str)
        .unique()
        .tolist()
    )

    input_gene_count = len(input_genes)
    single_query_mode = input_gene_count == 1

    sim_df = compute_similarity_from_gene_set_df(
        query_gene_set_df=query_gene_set_df,
        start_time=start_time,
        end_time=end_time,
        start_space=start_space,
        end_space=end_space
    )

    top_compare = make_top_compare(sim_df, int(top_n))

    input_gene_map = query_gene_set_df[
        ["locus_ID", "Gene_label", "category", "Color"]
    ].drop_duplicates()

    top_genes = get_top_genes(top_compare, top_method)
    top_genes_only = [g for g in top_genes if g not in input_genes]

    selected_genes = list(dict.fromkeys(input_genes + top_genes_only))

    df2 = df[
        (df["Gene"].isin(selected_genes)) &
        (df["Time"].isin(timepoints)) &
        (df["Space"].isin(spacepoints))
    ].copy()

    df2["Time_num"] = df2["Time"].map({t: i + 1 for i, t in enumerate(timepoints)})
    df2["Space_num"] = df2["Space"].map({s: i + 1 for i, s in enumerate(spacepoints)})

    df2["gene_group"] = np.where(
        df2["Gene"].isin(input_genes),
        "Input gene",
        "Closest top gene"
    )

    df2 = df2.merge(
        input_gene_map,
        left_on="Gene",
        right_on="locus_ID",
        how="left"
    )

    df2["Gene_label"] = np.where(
        df2["gene_group"] == "Input gene",
        df2["Gene_label"].fillna(df2["Gene"]),
        df2["Gene"]
    )

    df2 = df2.merge(ann, on="Gene", how="left")
    df2["GeneName"] = df2["GeneName"].fillna(df2["Gene"])

    query_df = df[
        (df["Gene"].isin(input_genes)) &
        (df["Time"].isin(timepoints)) &
        (df["Space"].isin(spacepoints))
    ].copy()

    query_df["Time_num"] = query_df["Time"].map({t: i + 1 for i, t in enumerate(timepoints)})
    query_df["Space_num"] = query_df["Space"].map({s: i + 1 for i, s in enumerate(spacepoints)})

    if single_query_mode:
        query_curve_df = query_df.copy()
        query_curve_label = str(query_gene_set_df["Gene_label"].iloc[0])
    else:
        query_curve_df = query_df.copy()
        query_curve_label = "Median query pattern"

    median_color = "#2ca02c"
    if "Color" in query_gene_set_df.columns and query_gene_set_df["Color"].notna().any():
        median_color = str(query_gene_set_df["Color"].dropna().iloc[0])

    query_summary = build_query_summary(
        query_gene_set_df=query_gene_set_df,
        missing_genes=missing_genes,
        query_status=query_status,
        single_query_mode=single_query_mode
    )

    return (
        df2,
        query_df,
        query_curve_df,
        sim_df,
        top_compare,
        timepoints,
        spacepoints,
        median_color,
        missing_genes,
        input_gene_count,
        single_query_mode,
        query_curve_label,
        query_status,
        query_gene_set_df,
        query_summary
    )


def make_temporal_figure(
    df2,
    query_df,
    query_curve_df,
    timepoints,
    spacepoints,
    median_color,
    fit_method,
    show_input,
    input_gene_count,
    single_query_mode,
    query_curve_label
):
    n_panels = len(spacepoints)
    ncols = min(4, n_panels)
    nrows = math.ceil(n_panels / ncols)

    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        subplot_titles=spacepoints,
        horizontal_spacing=0.05,
        vertical_spacing=0.16
    )

    show_legend_top = True
    show_legend_query = True

    show_input_gene_names_in_legend = show_input and (input_gene_count <= 10) and (not single_query_mode)

    input_gene_legend_seen = set()

    for idx, space in enumerate(spacepoints):
        row = idx // ncols + 1
        col = idx % ncols + 1

        space_df = df2[df2["Space"] == space]

        top_df = space_df[space_df["gene_group"] == "Closest top gene"]

        for gene, gdf in top_df.groupby("Gene"):
            gene_name = str(gdf["GeneName"].iloc[0]) if "GeneName" in gdf.columns else str(gene)

            xfit, yfit = fit_curve(
                gdf["Time_num"],
                gdf["logFC"],
                method=fit_method
            )

            if len(xfit) == 0:
                continue

            fig.add_trace(
                go.Scatter(
                    x=xfit,
                    y=yfit,
                    mode="lines",
                    line=dict(color="rgba(90,90,90,0.55)", width=1.6),
                    name="Closest top genes",
                    legendgroup="Closest top genes",
                    showlegend=show_legend_top,
                    hovertemplate=(
                        f"GeneName: {gene_name}<br>"
                        f"Gene ID: {gene}<br>"
                        f"Space: {space}<br>"
                        "logFC: %{y:.3f}<extra></extra>"
                    ),
                ),
                row=row,
                col=col
            )

            show_legend_top = False

        if show_input and (not single_query_mode):
            input_df = space_df[space_df["gene_group"] == "Input gene"]

            for label, gdf in input_df.groupby("Gene_label"):
                xfit, yfit = fit_curve(
                    gdf["Time_num"],
                    gdf["logFC"],
                    method=fit_method
                )

                if len(xfit) == 0:
                    continue

                if show_input_gene_names_in_legend:
                    trace_name = label
                    legend_group = f"input_{label}"
                    show_legend_flag = label not in input_gene_legend_seen
                    input_gene_legend_seen.add(label)
                else:
                    trace_name = "Input genes"
                    legend_group = "Input genes"
                    show_legend_flag = "Input genes" not in input_gene_legend_seen
                    input_gene_legend_seen.add("Input genes")

                fig.add_trace(
                    go.Scatter(
                        x=xfit,
                        y=yfit,
                        mode="lines",
                        line=dict(width=2.0),
                        name=trace_name,
                        legendgroup=legend_group,
                        showlegend=show_legend_flag,
                        hovertemplate=(
                            f"Input gene: {label}<br>"
                            f"Space: {space}<br>"
                            "logFC: %{y:.3f}<extra></extra>"
                        ),
                    ),
                    row=row,
                    col=col
                )

        if single_query_mode:
            qdf = (
                query_curve_df[query_curve_df["Space"] == space]
                .sort_values("Time_num")
            )
            x_source = qdf["Time_num"]
            y_source = qdf["logFC"]
        else:
            qdf = (
                query_curve_df[query_curve_df["Space"] == space]
                .groupby(["Time", "Time_num"], as_index=False)["logFC"]
                .median()
                .sort_values("Time_num")
            )
            x_source = qdf["Time_num"]
            y_source = qdf["logFC"]

        xfit, yfit = fit_curve(
            x_source,
            y_source,
            method=fit_method
        )

        fig.add_trace(
            go.Scatter(
                x=xfit,
                y=yfit,
                mode="lines",
                line=dict(color=median_color, width=4),
                name=query_curve_label,
                legendgroup=query_curve_label,
                showlegend=show_legend_query,
                hovertemplate=(
                    f"{query_curve_label}<br>"
                    f"Space: {space}<br>"
                    "logFC: %{y:.3f}<extra></extra>"
                ),
            ),
            row=row,
            col=col
        )

        show_legend_query = False

        fig.update_xaxes(
            tickmode="array",
            tickvals=list(range(1, len(timepoints) + 1)),
            ticktext=timepoints,
            row=row,
            col=col
        )

    fig.update_layout(
        template="plotly_white",
        height=max(520, 360 * nrows),
        title="Temporal fitted curves",
        font=dict(family="Arial", size=13),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=40, r=40, t=90, b=70)
    )

    fig.update_yaxes(title_text="logFC")
    fig.update_xaxes(title_text="Time")

    return fig


def make_spatial_figure(
    df2,
    query_df,
    query_curve_df,
    timepoints,
    spacepoints,
    median_color,
    fit_method,
    show_input,
    input_gene_count,
    single_query_mode,
    query_curve_label
):
    n_panels = len(timepoints)
    ncols = min(3, n_panels)
    nrows = math.ceil(n_panels / ncols)

    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        subplot_titles=timepoints,
        horizontal_spacing=0.06,
        vertical_spacing=0.24
    )

    show_legend_top = True
    show_legend_query = True

    show_input_gene_names_in_legend = show_input and (input_gene_count <= 10) and (not single_query_mode)

    input_gene_legend_seen = set()

    for idx, time in enumerate(timepoints):
        row = idx // ncols + 1
        col = idx % ncols + 1

        time_df = df2[df2["Time"] == time]

        top_df = time_df[time_df["gene_group"] == "Closest top gene"]

        for gene, gdf in top_df.groupby("Gene"):
            gene_name = str(gdf["GeneName"].iloc[0]) if "GeneName" in gdf.columns else str(gene)

            xfit, yfit = fit_curve(
                gdf["Space_num"],
                gdf["logFC"],
                method=fit_method
            )

            if len(xfit) == 0:
                continue

            fig.add_trace(
                go.Scatter(
                    x=xfit,
                    y=yfit,
                    mode="lines",
                    line=dict(color="rgba(90,90,90,0.55)", width=1.6),
                    name="Closest top genes",
                    legendgroup="Closest top genes",
                    showlegend=show_legend_top,
                    hovertemplate=(
                        f"GeneName: {gene_name}<br>"
                        f"Gene ID: {gene}<br>"
                        f"Time: {time}<br>"
                        "logFC: %{y:.3f}<extra></extra>"
                    ),
                ),
                row=row,
                col=col
            )

            show_legend_top = False

        if show_input and (not single_query_mode):
            input_df = time_df[time_df["gene_group"] == "Input gene"]

            for label, gdf in input_df.groupby("Gene_label"):
                xfit, yfit = fit_curve(
                    gdf["Space_num"],
                    gdf["logFC"],
                    method=fit_method
                )

                if len(xfit) == 0:
                    continue

                if show_input_gene_names_in_legend:
                    trace_name = label
                    legend_group = f"input_{label}"
                    show_legend_flag = label not in input_gene_legend_seen
                    input_gene_legend_seen.add(label)
                else:
                    trace_name = "Input genes"
                    legend_group = "Input genes"
                    show_legend_flag = "Input genes" not in input_gene_legend_seen
                    input_gene_legend_seen.add("Input genes")

                fig.add_trace(
                    go.Scatter(
                        x=xfit,
                        y=yfit,
                        mode="lines",
                        line=dict(width=2.0),
                        name=trace_name,
                        legendgroup=legend_group,
                        showlegend=show_legend_flag,
                        hovertemplate=(
                            f"Input gene: {label}<br>"
                            f"Time: {time}<br>"
                            "logFC: %{y:.3f}<extra></extra>"
                        ),
                    ),
                    row=row,
                    col=col
                )

        if single_query_mode:
            qdf = (
                query_curve_df[query_curve_df["Time"] == time]
                .sort_values("Space_num")
            )
            x_source = qdf["Space_num"]
            y_source = qdf["logFC"]
        else:
            qdf = (
                query_curve_df[query_curve_df["Time"] == time]
                .groupby(["Space", "Space_num"], as_index=False)["logFC"]
                .median()
                .sort_values("Space_num")
            )
            x_source = qdf["Space_num"]
            y_source = qdf["logFC"]

        xfit, yfit = fit_curve(
            x_source,
            y_source,
            method=fit_method
        )

        fig.add_trace(
            go.Scatter(
                x=xfit,
                y=yfit,
                mode="lines",
                line=dict(color=median_color, width=4),
                name=query_curve_label,
                legendgroup=query_curve_label,
                showlegend=show_legend_query,
                hovertemplate=(
                    f"{query_curve_label}<br>"
                    f"Time: {time}<br>"
                    "logFC: %{y:.3f}<extra></extra>"
                ),
            ),
            row=row,
            col=col
        )

        show_legend_query = False

        fig.update_xaxes(
            tickmode="array",
            tickvals=list(range(1, len(spacepoints) + 1)),
            ticktext=spacepoints,
            tickangle=45,
            row=row,
            col=col
        )

    fig.update_layout(
        template="plotly_white",
        height=max(520, 420 * nrows),
        title="Spatial fitted curves",
        font=dict(family="Arial", size=13),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.04,
            xanchor="right",
            x=1
        ),
        margin=dict(l=40, r=40, t=90, b=120)
    )

    fig.update_yaxes(title_text="logFC")
    fig.update_xaxes(title_text="Space")

    return fig


# ============================================================
# 5. Dash layout
# ============================================================

available_gene_sets_text = ", ".join(GENE_SET_FILES.keys())

layout = dbc.Container(
    [
        html.H2("Similarity Profile", className="page-title"),

        html.P(
            "Interactive Python/Plotly version of the spatial-temporal similarity visualization. "
            "By default, the app uses the predefined motV gene set. "
            "Users can also search any gene of interest by gene name or gene ID, or upload a gene list. "
            "If the query has only one gene, that gene is directly used as the query pattern. "
            "If the query has 2 or more genes, the median pattern is used.",
            className="lead"
        ),

        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Label("Query source"),
                        dcc.RadioItems(
                            id="similarity-query-source",
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
                        html.Label("Search gene name, gene ID, or predefined gene set"),
                        dcc.Dropdown(
                            id="similarity-query-text",
                            options=load_gene_lookup_options(),
                            value=["__gene_set__::motV"],
                            multi=True,
                            searchable=True,
                            clearable=True,
                            placeholder=(
                                "Type GeneName or Gene ID, e.g. motV or N900_RS00010"
                            ),
                        ),
                        html.Small(
                            f"Start typing to search GeneName/Gene ID. Available predefined gene sets: {available_gene_sets_text}",
                            className="text-muted",
                        ),
                        html.Div(id="similarity-query-summary"),
                    ],
                    md=5,
                ),
                dbc.Col(
                    [
                        html.Label("Start time"),
                        dcc.Dropdown(
                            id="similarity-start-time",
                            options=[{"label": t, "value": t} for t in FULL_TIMEPOINTS],
                            value="1h",
                            clearable=False,
                        ),
                    ],
                    md=2,
                ),
                dbc.Col(
                    [
                        html.Label("End time"),
                        dcc.Dropdown(
                            id="similarity-end-time",
                            options=[{"label": t, "value": t} for t in FULL_TIMEPOINTS],
                            value="24h",
                            clearable=False,
                        ),
                    ],
                    md=2,
                ),
            ],
            className="mb-3",
        ),

        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Label("Start space"),
                        dcc.Dropdown(
                            id="similarity-start-space",
                            options=[{"label": s, "value": s} for s in SPACEPOINTS],
                            value="st",
                            clearable=False,
                        ),
                    ],
                    md=2,
                ),
                dbc.Col(
                    [
                        html.Label("End space"),
                        dcc.Dropdown(
                            id="similarity-end-space",
                            options=[{"label": s, "value": s} for s in SPACEPOINTS],
                            value="co",
                            clearable=False,
                        ),
                    ],
                    md=2,
                ),
                dbc.Col(
                    [
                        html.Label("Top N most similar genes"),
                        dbc.Input(
                            id="similarity-top-n",
                            type="number",
                            min=1,
                            max=500,
                            step=1,
                            value=10,
                            placeholder="Enter Top N most similar genes",
                        ),
                    ],
                    md=3,
                ),
                dbc.Col(
                    [
                        html.Label("Top method"),
                        dcc.Dropdown(
                            id="similarity-top-method",
                            options=[
                                {"label": "DTW", "value": "DTW"},
                                {"label": "Cosine", "value": "cosine"},
                                {"label": "Both", "value": "both"},
                                {"label": "Union", "value": "union"},
                            ],
                            value="DTW",
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
                        html.Label("Upload gene list"),
                        dcc.Upload(
                            id="similarity-upload-gene-list",
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
                            id="similarity-upload-status",
                            className="text-muted mt-2"
                        ),
                    ],
                    md=6,
                ),
            ],
            className="mb-4",
        ),

        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Label("Curve type"),
                        dcc.RadioItems(
                            id="similarity-curve-type",
                            options=[
                                {"label": "Temporal", "value": "temporal"},
                                {"label": "Spatial", "value": "spatial"},
                            ],
                            value="temporal",
                            inline=True,
                        ),
                    ],
                    md=3,
                ),
                dbc.Col(
                    [
                        html.Label("Fit method"),
                        dcc.Dropdown(
                            id="similarity-fit-method",
                            options=[
                                {"label": "Polynomial degree 2", "value": "poly2"},
                                {"label": "Polynomial degree 3", "value": "poly3"},
                                {"label": "LOESS", "value": "loess"},
                                {"label": "Raw connected line", "value": "raw"},
                            ],
                            value="poly2",
                            clearable=False,
                        ),
                    ],
                    md=3,
                ),
                dbc.Col(
                    [
                        html.Label("Display options"),
                        dcc.Checklist(
                            id="similarity-display-options",
                            options=[
                                {"label": "Show input genes", "value": "show_input"},
                            ],
                            value=[],
                            inline=True,
                        ),
                    ],
                    md=3,
                ),
            ],
            className="mb-4",
        ),

        dbc.Alert(
            "Grey curves = closest top genes. Thick colored curve = query pattern. "
            "If only one query gene is provided, that gene itself is used as the query pattern. "
            "If Show input genes is enabled and the number of input genes is ≤10, the legend will display input gene names.",
            color="secondary",
            className="mb-4"
        ),

        dcc.Loading(
            id="similarity-loading",
            type="circle",
            children=[
                dcc.Graph(id="similarity-plot", style={"width": "100%"})
            ],
        ),

        html.Hr(),

        html.H4("Top closest genes"),

        dbc.Button(
            "Download top closest genes table",
            id="download-top-table-button",
            color="secondary",
            size="sm",
            className="mb-2"
        ),
        dcc.Download(id="download-top-table"),

        html.Div(id="similarity-top-table", className="mb-5"),

        html.H4("Full similarity ranking"),

        dbc.Button(
            "Download full similarity ranking table",
            id="download-full-table-button",
            color="secondary",
            size="sm",
            className="mb-2"
        ),
        dcc.Download(id="download-full-table"),

        html.Div(id="similarity-full-table", className="mb-5"),
    ],
    fluid=True
)


# ============================================================
# 6. Main callback
# ============================================================

@dash.callback(
    Output("similarity-plot", "figure"),
    Output("similarity-top-table", "children"),
    Output("similarity-full-table", "children"),
    Output("similarity-upload-status", "children"),
    Output("similarity-query-summary", "children"),
    Input("similarity-query-source", "value"),
    Input("similarity-query-text", "value"),
    Input("similarity-start-time", "value"),
    Input("similarity-end-time", "value"),
    Input("similarity-start-space", "value"),
    Input("similarity-end-space", "value"),
    Input("similarity-top-n", "value"),
    Input("similarity-top-method", "value"),
    Input("similarity-curve-type", "value"),
    Input("similarity-fit-method", "value"),
    Input("similarity-display-options", "value"),
    Input("similarity-upload-gene-list", "contents"),
    State("similarity-upload-gene-list", "filename"),
)
def update_similarity_page(
    query_source,
    query_text,
    start_time,
    end_time,
    start_space,
    end_space,
    top_n,
    top_method,
    curve_type,
    fit_method,
    display_options,
    upload_contents,
    upload_filename
):
    try:
        show_input = "show_input" in display_options

        top_n = int(top_n) if top_n is not None else 10
        top_n = max(1, min(top_n, 500))

        (
            df2,
            query_df,
            query_curve_df,
            sim_df,
            top_compare,
            timepoints,
            spacepoints,
            median_color,
            missing_genes,
            input_gene_count,
            single_query_mode,
            query_curve_label,
            query_status,
            query_gene_set_df,
            query_summary
        ) = prepare_plot_data(
            query_source=query_source,
            query_text=query_text,
            upload_contents=upload_contents,
            upload_filename=upload_filename,
            start_time=start_time,
            end_time=end_time,
            start_space=start_space,
            end_space=end_space,
            top_n=top_n,
            top_method=top_method,
        )

        if curve_type == "temporal":
            fig = make_temporal_figure(
                df2=df2,
                query_df=query_df,
                query_curve_df=query_curve_df,
                timepoints=timepoints,
                spacepoints=spacepoints,
                median_color=median_color,
                fit_method=fit_method,
                show_input=show_input,
                input_gene_count=input_gene_count,
                single_query_mode=single_query_mode,
                query_curve_label=query_curve_label
            )
        else:
            fig = make_spatial_figure(
                df2=df2,
                query_df=query_df,
                query_curve_df=query_curve_df,
                timepoints=timepoints,
                spacepoints=spacepoints,
                median_color=median_color,
                fit_method=fit_method,
                show_input=show_input,
                input_gene_count=input_gene_count,
                single_query_mode=single_query_mode,
                query_curve_label=query_curve_label
            )

        upload_status = query_status

        if single_query_mode:
            upload_status += " Single query gene detected: using that gene directly as the query pattern."
        else:
            upload_status += f" {input_gene_count} query genes detected: using the median pattern."

        if missing_genes:
            upload_status += f" Missing/unmatched genes: {', '.join(missing_genes[:10])}"
            if len(missing_genes) > 10:
                upload_status += f" ... and {len(missing_genes) - 10} more."

        top_table = dash_table.DataTable(
            data=top_compare.to_dict("records"),
            columns=[{"name": c, "id": c} for c in top_compare.columns],
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
                "maxWidth": "260px",
                "whiteSpace": "normal",
            },
            style_header={
                "fontWeight": "bold",
                "backgroundColor": "#f1f3f5",
            },
        )

        preferred_cols = ["Gene", "GeneName", "DTW_dist", "Cosine_dist"]
        cols = [c for c in preferred_cols if c in sim_df.columns]
        remaining = [c for c in sim_df.columns if c not in cols]
        sim_df_show = sim_df[cols + remaining].copy()

        full_table = dash_table.DataTable(
            data=sim_df_show.head(500).to_dict("records"),
            columns=[{"name": c, "id": c} for c in sim_df_show.columns],
            page_size=20,
            filter_action="native",
            sort_action="native",
            style_table={"overflowX": "auto"},
            style_cell={
                "textAlign": "left",
                "fontFamily": "Arial",
                "fontSize": "14px",
                "padding": "6px",
                "minWidth": "120px",
                "maxWidth": "260px",
                "whiteSpace": "normal",
            },
            style_header={
                "fontWeight": "bold",
                "backgroundColor": "#f1f3f5",
            },
        )

        return fig, top_table, full_table, upload_status, query_summary

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
                    font=dict(size=14)
                )
            ],
        )

        alert = dbc.Alert(str(e), color="danger")

        query_summary = dbc.Alert(
            [
                html.Strong("Query not recognized. "),
                "The input you typed was not found in the annotation table or raw data. ",
                "Please type a valid gene ID, try another gene name, or use a predefined gene set name such as motV."
            ],
            color="warning",
            className="mt-2"
        )

        return fig, alert, alert, "", query_summary


# ============================================================
# 7. Download callbacks
# ============================================================

def get_current_tables_for_download(
    query_source,
    query_text,
    start_time,
    end_time,
    start_space,
    end_space,
    top_n,
    top_method,
    upload_contents,
    upload_filename
):
    top_n = int(top_n) if top_n is not None else 10
    top_n = max(1, min(top_n, 500))

    (
        df2,
        query_df,
        query_curve_df,
        sim_df,
        top_compare,
        timepoints,
        spacepoints,
        median_color,
        missing_genes,
        input_gene_count,
        single_query_mode,
        query_curve_label,
        query_status,
        query_gene_set_df,
        query_summary
    ) = prepare_plot_data(
        query_source=query_source,
        query_text=query_text,
        upload_contents=upload_contents,
        upload_filename=upload_filename,
        start_time=start_time,
        end_time=end_time,
        start_space=start_space,
        end_space=end_space,
        top_n=top_n,
        top_method=top_method,
    )

    return top_compare, sim_df


@dash.callback(
    Output("download-top-table", "data"),
    Input("download-top-table-button", "n_clicks"),
    State("similarity-query-source", "value"),
    State("similarity-query-text", "value"),
    State("similarity-start-time", "value"),
    State("similarity-end-time", "value"),
    State("similarity-start-space", "value"),
    State("similarity-end-space", "value"),
    State("similarity-top-n", "value"),
    State("similarity-top-method", "value"),
    State("similarity-upload-gene-list", "contents"),
    State("similarity-upload-gene-list", "filename"),
    prevent_initial_call=True
)
def download_top_table(
    n_clicks,
    query_source,
    query_text,
    start_time,
    end_time,
    start_space,
    end_space,
    top_n,
    top_method,
    upload_contents,
    upload_filename
):
    if not n_clicks:
        return no_update

    top_compare, sim_df = get_current_tables_for_download(
        query_source=query_source,
        query_text=query_text,
        start_time=start_time,
        end_time=end_time,
        start_space=start_space,
        end_space=end_space,
        top_n=top_n,
        top_method=top_method,
        upload_contents=upload_contents,
        upload_filename=upload_filename
    )

    filename = (
        f"top_closest_genes_"
        f"{start_time}_to_{end_time}_"
        f"{start_space}_to_{end_space}_"
        f"Top_{top_n}.csv"
    )

    return dcc.send_data_frame(
        top_compare.to_csv,
        filename,
        index=False
    )


@dash.callback(
    Output("download-full-table", "data"),
    Input("download-full-table-button", "n_clicks"),
    State("similarity-query-source", "value"),
    State("similarity-query-text", "value"),
    State("similarity-start-time", "value"),
    State("similarity-end-time", "value"),
    State("similarity-start-space", "value"),
    State("similarity-end-space", "value"),
    State("similarity-top-n", "value"),
    State("similarity-top-method", "value"),
    State("similarity-upload-gene-list", "contents"),
    State("similarity-upload-gene-list", "filename"),
    prevent_initial_call=True
)
def download_full_table(
    n_clicks,
    query_source,
    query_text,
    start_time,
    end_time,
    start_space,
    end_space,
    top_n,
    top_method,
    upload_contents,
    upload_filename
):
    if not n_clicks:
        return no_update

    top_compare, sim_df = get_current_tables_for_download(
        query_source=query_source,
        query_text=query_text,
        start_time=start_time,
        end_time=end_time,
        start_space=start_space,
        end_space=end_space,
        top_n=top_n,
        top_method=top_method,
        upload_contents=upload_contents,
        upload_filename=upload_filename
    )

    filename = (
        f"full_similarity_ranking_"
        f"{start_time}_to_{end_time}_"
        f"{start_space}_to_{end_space}.csv"
    )

    return dcc.send_data_frame(
        sim_df.to_csv,
        filename,
        index=False
    )
