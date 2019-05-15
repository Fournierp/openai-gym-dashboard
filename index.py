# -*- coding: utf-8 -*-
import flask
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from app import app, server


app.layout = html.Div([
    html.Div([
        html.H2(
            'OpenAI Gym Dashboard',
            id='title'
        )
    ],

    # dcc.Graph(id='reward-graph'),
    #
    # dcc.Graph(id='loss-graph'),

    ),
])


if __name__ == "__main__":
    app.run_server(debug=True)
