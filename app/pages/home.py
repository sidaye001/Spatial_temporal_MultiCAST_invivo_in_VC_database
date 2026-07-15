import os
import base64

import dash
from dash import html
import dash_bootstrap_components as dbc


DATABASE_NAME = "Spatial-Temporal Gene Fitness Database in vivo for Vibrio cholerae"


dash.register_page(__name__, path="/", name="Home")


# ============================================================
# 0. Paths
# ============================================================

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(APP_DIR)

HOME_FIGURE_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "figure",
    "Home.png"
)


# ============================================================
# 1. Helper functions
# ============================================================

def encode_image(image_path):
    if not os.path.exists(image_path):
        return None

    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")

    return f"data:image/png;base64,{encoded}"


def module_item(title, text, href):
    return dbc.Col(
        dbc.Card(
            dbc.CardBody(
                [
                    html.H5(title, className="mb-2", style={"fontWeight": "650"}),
                    html.P(text, className="mb-3", style={"color": TEXT_COLOR}),
                    dbc.Button("Open", href=href, color="primary", outline=True, size="sm"),
                ]
            ),
            style=CARD_STYLE,
            className="h-100",
        ),
        lg=4,
        md=6,
        className="mb-3",
    )


HOME_IMAGE_SRC = encode_image(HOME_FIGURE_PATH)


# ============================================================
# 2. Styles
# ============================================================

TITLE_COLOR = "#123A7A"
TEXT_COLOR = "#3f4a54"
MUTED_COLOR = "#6c757d"

CARD_STYLE = {
    "borderRadius": "8px",
    "boxShadow": "0 2px 10px rgba(0,0,0,0.05)",
    "border": "1px solid rgba(18, 58, 122, 0.10)",
}

SECTION_TITLE_STYLE = {
    "fontWeight": "650",
    "color": TITLE_COLOR,
    "marginBottom": "0.85rem",
}


# ============================================================
# 3. Layout
# ============================================================

layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Img(
                            src=HOME_IMAGE_SRC,
                            style={
                                "width": "100%",
                                "display": "block",
                                "borderRadius": "8px",
                                "border": "1px solid rgba(18, 58, 122, 0.10)",
                            },
                        )
                        if HOME_IMAGE_SRC is not None
                        else dbc.Alert(
                            [
                                html.Strong("Home figure not found. "),
                                f"Expected file path: {HOME_FIGURE_PATH}",
                            ],
                            color="warning",
                            className="mb-0",
                        )
                    ],
                    width=12,
                )
            ],
            className="mb-4",
        ),

        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H4("What You Can Explore", style=SECTION_TITLE_STYLE),
                                html.P(
                                    "The dataset is organized around a spatial-temporal fitness matrix: "
                                    "each gene has measured fitness values across multiple infection "
                                    "time points and intestinal regions. The pages below expose different "
                                    "ways to inspect, compare, rank, and contextualize those profiles.",
                                    style={"color": TEXT_COLOR},
                                ),
                                html.Ul(
                                    [
                                        html.Li("Search genes by locus ID, gene name, or VC_ID."),
                                        html.Li("Inspect spatial-temporal landscapes for individual genes and gene sets."),
                                        html.Li("Find genes with correlated, clustered, or query-matched fitness behavior."),
                                        html.Li("Browse network relationships derived from cofitness structure."),
                                    ],
                                    style={"marginBottom": "0"},
                                ),
                            ]
                        ),
                        style={**CARD_STYLE, "height": "100%"},
                    ),
                    lg=7,
                    className="mb-4",
                ),
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H4("Query Identifiers", style=SECTION_TITLE_STYLE),
                                html.P("Supported gene search terms include:", className="mb-2"),
                                html.Ul(
                                    [
                                        html.Li("Locus IDs such as N900_RS04840"),
                                        html.Li("Gene names such as motV"),
                                        html.Li("VC_ID values such as VC_1909"),
                                    ],
                                    style={"marginBottom": "0.75rem"},
                                ),
                                html.P(
                                    "Pages with gene search boxes resolve these identifiers to the "
                                    "canonical locus ID used by the fitness matrix.",
                                    style={"color": MUTED_COLOR, "marginBottom": "0"},
                                ),
                            ]
                        ),
                        style={**CARD_STYLE, "height": "100%"},
                    ),
                    lg=5,
                    className="mb-4",
                ),
            ],
        ),

        html.H4("Analysis Pages", style=SECTION_TITLE_STYLE),
        dbc.Row(
            [
                module_item(
                    "Descriptive Fitness",
                    "Visualize single-gene fitness landscapes and summarize predefined, pasted, or uploaded gene sets across selected time and space windows.",
                    "/descriptive-fitness",
                ),
                module_item(
                    "Cofitness",
                    "Explore pairwise cofitness correlations, heatmaps, and top positively or negatively correlated partner genes.",
                    "/cofitness",
                ),
                module_item(
                    "Clustering",
                    "Inspect precomputed DTW and cosine clusters, view spatial or temporal fitness profiles, and download cluster assignments.",
                    "/clustering",
                ),
                module_item(
                    "Similarity Profile",
                    "Use one gene or a gene set as a query pattern to rank genes by DTW and cosine distance.",
                    "/similarity-profile",
                ),
                module_item(
                    "Network Browser",
                    "Place query genes in a GGM-based functional network and inspect query-connected subnetworks and ranked neighbors.",
                    "/network-browser",
                ),
                module_item(
                    "Predefined Pattern",
                    "Define a spatial-temporal trend and identify genes whose in vivo fitness profiles match that pattern.",
                    "/predefined-pattern",
                ),
            ],
            className="mb-4",
        ),

        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H4("AI Prediction", style=SECTION_TITLE_STYLE),
                                html.P(
                                    "The AI Prediction page is reserved for future predictive modules. "
                                    "The current public functionality is focused on interactive exploration, "
                                    "profile comparison, clustering, and network browsing.",
                                    style={"marginBottom": "0"},
                                ),
                            ]
                        ),
                        style=CARD_STYLE,
                    ),
                    md=6,
                    className="mb-4",
                ),
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H4("Citation", style=SECTION_TITLE_STYLE),
                                html.P(
                                    "If you use this database, please cite the corresponding publication or contact mwaldor at bwh dot harvard dot edu",
                                    style={"marginBottom": "0"},
                                ),
                            ]
                        ),
                        style=CARD_STYLE,
                    ),
                    md=6,
                    className="mb-4",
                ),
            ],
        ),

        html.Hr(),
        html.P(
            f"{DATABASE_NAME} is under active development; page content and analysis modules may be updated as the database evolves.",
            style={
                "textAlign": "center",
                "color": MUTED_COLOR,
                "marginBottom": "0",
            },
        ),
    ],
    fluid=True,
    style={
        "paddingTop": "1.5rem",
        "paddingBottom": "2rem",
        "maxWidth": "1600px",
    },
)
