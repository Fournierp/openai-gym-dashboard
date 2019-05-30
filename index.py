import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from plotly import tools

import pandas as pd

LOGFILE = ''

app = dash.Dash(__name__)
server = app.server


def div_graph(name):
    """
    Generates an html Div containing graph and control options for smoothing and display, given the name
    :param name:
    :return:
    """
    return html.Div([
        html.Div(
            id=f'div-{name}-graph',
            className="ten columns"
        ),

        html.Div([
            html.Div([
                html.P("Smoothing:", style={'font-weight': 'bold', 'margin-bottom': '0px'}),

                dcc.Checklist(
                    options=[
                        {'label': ' Training', 'value': 'train'},
                        {'label': ' Validation', 'value': 'val'}
                    ],
                    values=[],
                    id=f'checklist-smoothing-options-{name}'
                )
            ],
                style={'margin-top': '10px'}
            ),

            html.Div([
                dcc.Slider(
                    min=0,
                    max=1,
                    step=0.05,
                    marks={i / 5: i / 5 for i in range(0, 6)},
                    value=0.6,
                    updatemode='drag',
                    id=f'slider-smoothing-{name}'
                )
            ],
                style={'margin-bottom': '40px'}
            ),

            html.Div([
                html.P("Plot Display mode:", style={'font-weight': 'bold', 'margin-bottom': '0px'}),

                dcc.RadioItems(
                    options=[
                        {'label': ' Overlapping', 'value': 'overlap'},
                        {'label': ' Separate (Vertical)', 'value': 'separate_vertical'},
                        {'label': ' Separate (Horizontal)', 'value': 'separate_horizontal'}
                    ],
                    value='overlap',
                    id=f'radio-display-mode-{name}'
                ),

                html.Div(id=f'div-current-{name}-value')
            ]),
        ],
            className="two columns"
        ),
    ],
        className="row"
    )


# Content of the page
app.layout = html.Div([
    # Banner display
    html.Div([
        html.H2(
            'OpenAI Gym Dashboard',
            id='title'
        ),
        # html.Img(
        #     src="https://s3-us-west-1.amazonaws.com/plotly-tutorials/logo/new-branding/dash-logo-by-plotly-stripe-inverted.png"
        # )
    ],
        className="banner"
    ),

    # Body
    html.Div([
        html.Div([
            dcc.Dropdown(
                id='dropdown-interval-control',
                options=[
                    {'label': 'No Updates', 'value': 'no'},
                    {'label': 'Slow Updates', 'value': 'slow'},
                    {'label': 'Regular Updates', 'value': 'regular'},
                    {'label': 'Fast Updates', 'value': 'fast'}
                ],
                value='regular',
                className='ten columns',
                clearable=False,
                searchable=False
            ),

            html.Div(
                id="div-step-display",
                className="two columns"
            )
        ],
            id='div-interval-control',
            className='row'
        ),

        dcc.Interval(
            id="interval-log-update",
            n_intervals=0
        ),

        # Hidden Div Storing JSON-serialized DataFrame of run log
        html.Div(id='run-log-storage', style={'display': 'none'}),

        # The html divs storing the graphs and display parameters
        div_graph('reward'),
        div_graph('loss'),
    ],
        className="container"
    )
])


def update_graph(graph_id, graph_title, y_train_index, y_val_index, run_log_json, display_mode,
                 checklist_smoothing_options, slider_smoothing, yaxis_title):
    """
    Function that updates the Graphs with new data from the logs
    :param graph_id: ID for Dash callbacks
    :param graph_title: Title displayed on layout
    :param y_train_index: name of column index for y train we want to retrieve
    :param y_val_index: name of column index for y val we want to retrieve
    :param run_log_json: the json file containing the data
    :param display_mode: 'separate' or 'overlap'
    :param checklist_smoothing_options: 'train' or 'val'
    :param slider_smoothing: value between 0 and 1, at interval of 0.05
    :param yaxis_title:
    :return: dcc Graph object containing the updated figures
    """

    def smooth(scalars, weight=0.6):
        """
        Function to smooth the curve to display.
        :param scalars: Values of the graph
        :param weight: How much to smooth the curve by
        :return: smoothed values
        """
        last = scalars[0]
        smoothed = list()
        for point in scalars:
            smoothed_val = last * weight + (1 - weight) * point
            smoothed.append(smoothed_val)
            last = smoothed_val
        return smoothed

    if run_log_json:
        # Create graph
        layout = go.Layout(
            title=graph_title,
            margin=go.Margin(l=50, r=50, b=50, t=50),
            yaxis={'title': yaxis_title}
        )

        # Get the data from the json file stored in the page
        run_log_df = pd.read_json(run_log_json, orient='split')
        # Get the values for the curve
        step = run_log_df['Batch']
        y_train = run_log_df[y_train_index]
        y_val = run_log_df[y_val_index]

        # Apply Smoothing if needed
        if 'train' in checklist_smoothing_options:
            y_train = smooth(y_train, weight=slider_smoothing)

        if 'val' in checklist_smoothing_options:
            y_val = smooth(y_val, weight=slider_smoothing)

        # Draw the curves
        trace_train = go.Scatter(
            x=step,
            y=y_train,
            mode='lines',
            name='Training'
        )

        trace_val = go.Scatter(
            x=step,
            y=y_val,
            mode='lines',
            name='Validation'
        )

        if display_mode == 'separate_vertical':
            figure = tools.make_subplots(rows=2,
                                         cols=1,
                                         print_grid=False,
                                         shared_yaxes=True)

            figure.append_trace(trace_train, 1, 1)
            figure.append_trace(trace_val, 2, 1)

            figure['layout'].update(title=layout.title,
                                    margin=layout.margin,
                                    scene={'domain': {'x': (0., 0.5), 'y': (0.5, 1)}})

        elif display_mode == 'separate_horizontal':
            figure = tools.make_subplots(rows=1,
                                         cols=2,
                                         shared_yaxes=True,
                                         print_grid=False)

            figure.append_trace(trace_train, 1, 1)
            figure.append_trace(trace_val, 1, 2)

            figure['layout'].update(title=layout.title,
                                    margin=layout.margin)

        elif display_mode == 'overlap':
            figure = go.Figure(
                data=[trace_train, trace_val],
                layout=layout
            )

        else:
            figure = None

        return dcc.Graph(figure=figure, id=graph_id)

    return dcc.Graph(id=graph_id)


@app.callback(Output('interval-log-update', 'interval'),
              [Input('dropdown-interval-control', 'value')])
def update_interval_log_update(interval_rate):
    """
    Select the interval time between updates for the graph.
    :param interval_rate: user selection
    :return: interval rate
    """
    if interval_rate == 'fast':
        return 500

    elif interval_rate == 'regular':
        return 1000

    elif interval_rate == 'slow':
        return 5 * 1000

    # Refreshes every 24 hours
    elif interval_rate == 'no':
        return 24 * 60 * 60 * 1000


@app.callback(Output('run-log-storage', 'children'),
              [Input('interval-log-update', 'n_intervals')])
def get_run_log(_):
    """
    Function that gets the json file
    :param _:
    :return: data in json format
    """
    names = ['Batch', 'Reward', 'Loss']

    try:
        run_log_df = pd.read_csv(LOGFILE, names=names)
        json = run_log_df.to_json(orient='split')
    except FileNotFoundError as error:
        print(error)
        print("Please verify if the csv file generated by your model is placed in the correct directory.")
        return None

    return json


@app.callback(Output('div-step-display', 'children'),
              [Input('run-log-storage', 'children')])
def update_div_step_display(run_log_json):
    """
    Function that gets the last element from the JSON data.
    :param run_log_json: JSON log data
    :return: last row of the JSON log data
    """
    if run_log_json:
        run_log_df = pd.read_json(run_log_json, orient='split')
        return html.H6(f"Step: {run_log_df['step'].iloc[-1]}", style={'margin-top': '3px'})


@app.callback(Output('div-accuracy-graph', 'children'),
              [Input('run-log-storage', 'children'),
               Input('radio-display-mode-accuracy', 'value'),
               Input('checklist-smoothing-options-accuracy', 'values'),
               Input('slider-smoothing-accuracy', 'value')])
def update_accuracy_graph(run_log_json,
                          display_mode,
                          checklist_smoothing_options,
                          slider_smoothing):
    graph = update_graph('accuracy-graph',
                         'Prediction Accuracy',
                         'train accuracy',
                         'val accuracy',
                         run_log_json,
                         display_mode,
                         checklist_smoothing_options,
                         slider_smoothing,
                         'Accuracy')

    try:
        if display_mode in ['separate_horizontal', 'overlap']:
            graph.figure.layout.yaxis['range'] = [0, 1]
        else:
            graph.figure.layout.yaxis1['range'] = [0, 1]
            graph.figure.layout.yaxis2['range'] = [0, 1]

    except AttributeError:
        pass

    return [graph]


@app.callback(Output('div-cross-entropy-graph', 'children'),
              [Input('run-log-storage', 'children'),
               Input('radio-display-mode-cross-entropy', 'value'),
               Input('checklist-smoothing-options-cross-entropy', 'values'),
               Input('slider-smoothing-cross-entropy', 'value')])
def update_cross_entropy_graph(run_log_json,
                               display_mode,
                               checklist_smoothing_options,
                               slider_smoothing):
    graph = update_graph('cross-entropy-graph',
                         'Cross Entropy Loss',
                         'train cross entropy',
                         'val cross entropy',
                         run_log_json,
                         display_mode,
                         checklist_smoothing_options,
                         slider_smoothing,
                         'Loss')
    return [graph]


@app.callback(Output('div-current-accuracy-value', 'children'),
              [Input('run-log-storage', 'children')])
def update_div_current_accuracy_value(run_log_json):
    if run_log_json:
        run_log_df = pd.read_json(run_log_json, orient='split')
        return [
            html.P(
                "Current Accuracy:",
                style={
                    'font-weight': 'bold',
                    'margin-top': '15px',
                    'margin-bottom': '0px'
                }
            ),
            html.Div(f"Training: {run_log_df['train accuracy'].iloc[-1]:.4f}"),
            html.Div(f"Validation: {run_log_df['val accuracy'].iloc[-1]:.4f}")
        ]


@app.callback(Output('div-current-cross-entropy-value', 'children'),
              [Input('run-log-storage', 'children')])
def update_div_current_cross_entropy_value(run_log_json):
    if run_log_json:
        run_log_df = pd.read_json(run_log_json, orient='split')
        return [
            html.P(
                "Current Loss:",
                style={
                    'font-weight': 'bold',
                    'margin-top': '15px',
                    'margin-bottom': '0px'
                }
            ),
            html.Div(f"Training: {run_log_df['train cross entropy'].iloc[-1]:.4f}"),
            html.Div(f"Validation: {run_log_df['val cross entropy'].iloc[-1]:.4f}")
        ]

# Running the server
if __name__ == '__main__':
    app.run_server(debug=True)
