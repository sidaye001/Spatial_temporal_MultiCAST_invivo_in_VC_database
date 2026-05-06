import dash
from dash import html
import dash_bootstrap_components as dbc


dash.register_page(
    __name__,
    path="/ai-prediction",
    name="AI Prediction"
)


layout = dbc.Container(
    [
        html.H2("AI Prediction", className="page-title"),

        html.P(
            "This page will host future AI-based prediction modules for gene fitness, "
            "functional annotation, and environment-dependent phenotype prediction.",
            className="lead"
        ),

        dbc.Alert(
            [
                html.Strong("Placeholder page: "),
                "The AI prediction module is under development."
            ],
            color="info",
            className="mb-4",
        ),

        dbc.Card(
            dbc.CardBody(
                [
                    html.H4("Planned functions", className="card-title"),

                    html.Ul(
                        [
                            html.Li(
                                "Predict gene fitness profiles across time, space, or environmental conditions."
                            ),
                            html.Li(
                                "Use learned co-fitness or network features to infer candidate functional partners."
                            ),
                            html.Li(
                                "Integrate gene annotation, sequence features, and experimental fitness data."
                            ),
                            html.Li(
                                "Provide downloadable prediction tables and interactive visualizations."
                            ),
                        ]
                    ),
                ]
            ),
            className="mb-4",
        ),

        dbc.Card(
            dbc.CardBody(
                [
                    html.H4("Current status", className="card-title"),
                    html.P(
                        "No model is connected yet. This page is reserved for future AI prediction workflows."
                    ),
                ]
            )
        ),
    ],
    fluid=True
)
