import pathlib

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

PROJECT_DIR = pathlib.Path(__file__).resolve().parents[2]
FONT_COLOR = "#4A4B52"
BACKGROUND_COLOR = "#FFFCFA"
GRADIENT_COLOR = "#BAB8B8"
COLOR_SCHEME = ("#4A4B52", "#FCFCFC", "#E8BA91")
TICKSIZE = 11

# Set Plotly theme.
pio.templates["minimalist"] = go.layout.Template(
    layout=go.Layout(
        font_family="Open Sans",
        font_color=FONT_COLOR,
        title_font_size=20,
        plot_bgcolor=BACKGROUND_COLOR,
        paper_bgcolor=BACKGROUND_COLOR,
        xaxis=dict(tickfont_size=TICKSIZE, titlefont_size=TICKSIZE, showgrid=False),
        yaxis=dict(tickfont_size=TICKSIZE, titlefont_size=TICKSIZE, showgrid=False),
        width=840,
        height=540,
    ),
    layout_colorway=COLOR_SCHEME,
)
pio.templates.default = "plotly+minimalist"


def draw_learning_curves(history, fig_path):
    history = pd.DataFrame(history)
    history.index += 1  # Epoch starts from 1
    fig = px.line(
        history,
        labels={"value": "", "index": "Epoch", "variable": "Variable"},
        title="SE-ResNet - Training Process",
        line_dash="variable",
        line_dash_sequence=["solid", "solid", "dashdot", "dashdot"],
        color="variable",
        color_discrete_sequence=["#4A4B52", "#4A4B52", "#633075", "#633075"],
        height=480,
        width=840,
    )
    fig.update_traces(line_width=1.75, opacity=0.8)
    fig.update_xaxes(range=(-1, len(history["loss"])))
    fig.update_yaxes(range=(0.2, 0.95))
    fig.write_image(fig_path)
