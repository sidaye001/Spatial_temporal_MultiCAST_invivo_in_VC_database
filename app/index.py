import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
from app import app, server

navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Home", href="/")),
        dbc.NavItem(dbc.NavLink("Descriptive Fitness", href="/descriptive-fitness")),
        dbc.NavItem(dbc.NavLink("Cofitness", href="/cofitness")),
        dbc.NavItem(dbc.NavLink("Clustering", href="/clustering")),
        dbc.NavItem(dbc.NavLink("Similarity Profile", href="/similarity-profile")),
        dbc.NavItem(dbc.NavLink("Network Browser", href="/network-browser")),
        dbc.NavItem(dbc.NavLink("Predefined Pattern", href="/predefined-pattern")),
        dbc.NavItem(dbc.NavLink("AI Prediction", href="/ai-prediction")),
    ],
    brand="Spatial-Temporal in vivo Fitness Database for Vibrio Cholerae",
    brand_href="/",
    color="primary",
    dark=True,
    fluid=True,
)

app.layout = html.Div(
    [
        dcc.Location(id="url"),
        navbar,
        dbc.Container(
            dash.page_container,
            fluid=True,
            className="main-container"
        ),
    ]
)

if __name__ == "__main__":
    app.run_server(
        debug=True,
        host="127.0.0.1",
        port=8050
    )
