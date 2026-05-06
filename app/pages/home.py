import os
import base64

import dash
from dash import html
import dash_bootstrap_components as dbc


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
    """
    Encode a local image file as base64 so Dash can display it directly.
    """
    if not os.path.exists(image_path):
        return None

    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")

    return f"data:image/png;base64,{encoded}"


HOME_IMAGE_SRC = encode_image(HOME_FIGURE_PATH)


# ============================================================
# 2. Reusable styles
# ============================================================

TITLE_COLOR = "#123A7A"
TEXT_COLOR = "#4a4a4a"
MUTED_COLOR = "#6c757d"

CARD_STYLE = {
    "borderRadius": "14px",
    "boxShadow": "0 2px 10px rgba(0,0,0,0.06)",
    "border": "1px solid rgba(18, 58, 122, 0.08)",
}

SECTION_TITLE_STYLE = {
    "fontWeight": "600",
    "color": TITLE_COLOR,
    "marginBottom": "0.9rem",
}


# ============================================================
# 3. Layout
# ============================================================

layout = dbc.Container(
    [
        # ======================================================
        # Hero title
        # ======================================================
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H1(
                            "Spatial-Temporal Gene Fitness Database in vivo for Vibrio cholerae",
                            style={
                                "fontWeight": "700",
                                "color": TITLE_COLOR,
                                "marginBottom": "0.5rem",
                                "lineHeight": "1.15",
                            },
                        ),
                        html.P(
                            "An interactive public web database for exploring in vivo gene fitness "
                            "patterns of Vibrio cholerae across intestinal space, infection time, "
                            "and functional relationships.",
                            style={
                                "fontSize": "1.15rem",
                                "color": TEXT_COLOR,
                                "marginBottom": "0.5rem",
                            },
                        ),
                        html.P(
                            "This resource organizes spatial-temporal in vivo fitness measurements into "
                            "an interactive analysis platform, enabling users to explore descriptive "
                            "fitness trends, cofitness, clustering, similarity profiles, network "
                            "relationships, and future AI-based prediction.",
                            style={
                                "fontSize": "1.02rem",
                                "color": "#5a5a5a",
                                "marginBottom": "0",
                            },
                        ),
                    ],
                    width=12,
                )
            ],
            className="mb-4",
        ),

        # ======================================================
        # Home schematic figure
        # ======================================================
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H4(
                                    "Database Overview",
                                    style=SECTION_TITLE_STYLE,
                                ),
                                html.Div(
                                    [
                                        html.Img(
                                            src=HOME_IMAGE_SRC,
                                            style={
                                                "width": "100%",
                                                "maxWidth": "1450px",
                                                "display": "block",
                                                "margin": "0 auto",
                                                "borderRadius": "10px",
                                            },
                                        )
                                    ]
                                    if HOME_IMAGE_SRC is not None
                                    else [
                                        dbc.Alert(
                                            [
                                                html.Strong("Home figure not found. "),
                                                f"Expected file path: {HOME_FIGURE_PATH}",
                                            ],
                                            color="warning",
                                            className="mb-0",
                                        )
                                    ]
                                ),
                            ]
                        ),
                        style=CARD_STYLE,
                    ),
                    width=12,
                )
            ],
            className="mb-4",
        ),

        # ======================================================
        # Introduction and contents
        # ======================================================
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H4(
                                    "Introduction",
                                    style=SECTION_TITLE_STYLE,
                                ),
                                html.P(
                                    "Understanding how bacterial gene fitness changes across host space "
                                    "and infection time is essential for dissecting infection biology. "
                                    "This database provides a structured and interactive framework to "
                                    "explore in vivo gene fitness dynamics for Vibrio cholerae."
                                ),
                                html.P(
                                    "The current database is built around a spatial-temporal in vivo "
                                    "fitness dataset, including multiple intestinal locations and time "
                                    "points. It is designed to help users examine how genes behave across "
                                    "intestinal niches, identify genes with related fitness patterns, "
                                    "define clusters of similar trajectories, and explore higher-order "
                                    "functional relationships."
                                ),
                                html.P(
                                    "This platform is intended for hypothesis generation, biological "
                                    "interpretation, and public data sharing. It also serves as a "
                                    "foundation for future predictive analysis modules.",
                                    style={"marginBottom": "0"},
                                ),
                            ]
                        ),
                        style={**CARD_STYLE, "height": "100%"},
                    ),
                    md=7,
                    className="mb-4",
                ),
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H4(
                                    "What this database contains",
                                    style=SECTION_TITLE_STYLE,
                                ),
                                html.Ul(
                                    [
                                        html.Li("Spatial in vivo fitness profiles across intestinal locations"),
                                        html.Li("Temporal in vivo fitness profiles across infection time points"),
                                        html.Li("Interactive cofitness analysis"),
                                        html.Li("Clustering of gene fitness trajectories"),
                                        html.Li("Similarity-based profile exploration"),
                                        html.Li("Network-based functional relationship browsing"),
                                        html.Li("Placeholder for AI-based prediction modules"),
                                    ],
                                    style={"marginBottom": "0"},
                                ),
                            ]
                        ),
                        style={**CARD_STYLE, "height": "100%"},
                    ),
                    md=5,
                    className="mb-4",
                ),
            ],
        ),

        # ======================================================
        # Database modules
        # ======================================================
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H4(
                                    "Database Analysis Modules",
                                    style=SECTION_TITLE_STYLE,
                                ),
                                html.P(
                                    "The database currently includes or is being expanded to include "
                                    "the following modules:"
                                ),
                                html.Ol(
                                    [
                                        html.Li(
                                            [
                                                html.Strong("Descriptive fitness"),
                                                " – explore overall gene fitness distributions and descriptive views.",
                                            ]
                                        ),
                                        html.Li(
                                            [
                                                html.Strong("Cofitness"),
                                                " – examine pairwise cofitness relationships and interactive heatmaps.",
                                            ]
                                        ),
                                        html.Li(
                                            [
                                                html.Strong("Clustering"),
                                                " – identify groups of genes with similar spatial, temporal, or global trajectories.",
                                            ]
                                        ),
                                        html.Li(
                                            [
                                                html.Strong("Similarity Profile"),
                                                " – find genes with the most similar fitness profiles to a query gene or gene set.",
                                            ]
                                        ),
                                        html.Li(
                                            [
                                                html.Strong("Network Browser"),
                                                " – explore query genes within the broader functional network.",
                                            ]
                                        ),
                                        html.Li(
                                            [
                                                html.Strong("AI Prediction"),
                                                " – placeholder for future machine learning and prediction modules.",
                                            ]
                                        ),
                                    ],
                                    style={"marginBottom": "0"},
                                ),
                            ]
                        ),
                        style=CARD_STYLE,
                    ),
                    width=12,
                )
            ],
            className="mb-4",
        ),

        # ======================================================
        # Usage guide
        # ======================================================
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H4(
                                    "How to use this database",
                                    style=SECTION_TITLE_STYLE,
                                ),
                                html.P(
                                    "Users can begin with the descriptive fitness module to examine global "
                                    "fitness patterns, then move to cofitness, clustering, similarity search, "
                                    "or network browsing for more targeted biological interpretation."
                                ),
                                html.P(
                                    "Gene-level queries can be performed using gene IDs or available gene names. "
                                    "Interactive plots support zooming, hovering, filtering, and downloadable "
                                    "tables where available.",
                                    style={"marginBottom": "0"},
                                ),
                            ]
                        ),
                        style=CARD_STYLE,
                    ),
                    width=12,
                )
            ],
            className="mb-4",
        ),

        # ======================================================
        # Citation and contact placeholders
        # ======================================================
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H4(
                                    "Citation",
                                    style=SECTION_TITLE_STYLE,
                                ),
                                html.P(
                                    "If you use this database in your work, please cite the corresponding "
                                    "publication(s).",
                                    className="mb-2",
                                ),
                                html.Div(
                                    [
                                        html.P(
                                            [
                                                html.Strong("[Placeholder citation 1] "),
                                                "Franz, Ye S., George et al. Spatial-Temporal in vivo Fitness "
                                                "for Vibrio cholerae reveal XXXXXX. Journal / preprint information will "
                                                "be added here.",
                                            ],
                                            style={"marginBottom": "0.5rem"},
                                        ),
                                        html.P(
                                            [
                                                html.Strong("[Placeholder citation 2] "),
                                                "Ye S., Franz et al. Database, companion manuscript, or methods citation here.",
                                            ],
                                            style={"marginBottom": "0"},
                                        ),
                                    ]
                                ),
                            ]
                        ),
                        style={**CARD_STYLE, "height": "100%"},
                    ),
                    md=6,
                    className="mb-4",
                ),
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H4(
                                    "Contact",
                                    style=SECTION_TITLE_STYLE,
                                ),
                                html.P(
                                    "For questions, bug reports, collaboration interests, or data-related "
                                    "inquiries, please contact:",
                                    className="mb-2",
                                ),
                                html.P(
                                    [
                                        html.Strong("[Placeholder name] "),
                                        "Matthew K. Waldor",
                                    ],
                                    style={"marginBottom": "0.35rem"},
                                ),
                                html.P(
                                    [
                                        html.Strong("[Placeholder email] "),
                                        "your_email@institution.edu",
                                    ],
                                    style={"marginBottom": "0.35rem"},
                                ),
                                html.P(
                                    [
                                        html.Strong("[Placeholder lab / institution] "),
                                        "Department / Lab / Institution",
                                    ],
                                    style={"marginBottom": "0"},
                                ),
                            ]
                        ),
                        style={**CARD_STYLE, "height": "100%"},
                    ),
                    md=6,
                    className="mb-4",
                ),
            ],
        ),

        # ======================================================
        # Footer note
        # ======================================================
        dbc.Row(
            [
                dbc.Col(
                    html.Div(
                        [
                            html.Hr(),
                            html.P(
                                "This website is under active development. Content, modules, citations, "
                                "and contact information will be updated as the database evolves.",
                                style={
                                    "textAlign": "center",
                                    "color": MUTED_COLOR,
                                    "marginTop": "0.5rem",
                                    "marginBottom": "0",
                                },
                            ),
                        ]
                    ),
                    width=12,
                )
            ]
        ),
    ],
    fluid=True,
    style={
        "paddingTop": "1.5rem",
        "paddingBottom": "2rem",
        "maxWidth": "1600px",
    },
)
