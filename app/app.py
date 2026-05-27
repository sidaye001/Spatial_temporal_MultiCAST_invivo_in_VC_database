import dash
import dash_bootstrap_components as dbc

app = dash.Dash(
    __name__,
    use_pages=True,
    external_stylesheets=[dbc.themes.FLATLY],
    suppress_callback_exceptions=True,
    title="Spatial-Temporal Gene Fitness Database in vivo for Vibrio cholerae"
)

server = app.server
