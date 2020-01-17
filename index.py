import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go

import pandas as pd
import numpy as np

from os import listdir

# Global variables for web-app functions
LOGFILE = ''

app = dash.Dash(__name__)
server = app.server


def div_graph(name):
    """
    Generates unique html Div containing graph and control options for smoothing, given the name.
    :param name: name of the div
    :return: html Div
    """
    return html.Div([
        html.Div(
            id=f'div-{name}-graph',
            className="ten columns"
        ),

        html.Div([
            html.Div([
                html.P("Smoothing:", style={'font-weight': 'bold', 'margin-bottom': '0px'}),

                dcc.RadioItems(
                    options=[
                        {'label': ' Yes', 'value': 'yes'},
                        {'label': ' No', 'value': 'no'}
                    ],
                    value='no',
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

            html.Div(id=f'div-current-{name}-value'),

        ],
            className="two columns"
        ),
    ],
        className="row"
    )

def get_log_files(type):
    """
    Function that fetches all the files in the logs directory to display in the dropdown.
    :return: list of file paths
    """
    options = []

    if( type == "live" ):
        # Get logs of models being trained live
        for log in listdir('agents/logs/live'):
            new_log = {'label': log, 'value': '/'.join(('agents/logs/live', log))}
            options.append(new_log)
    else:
        # Get logs of models already trained
        for log in listdir('agents/logs/completed'):
            new_log = {'label': log, 'value': '/'.join(('agents/logs/completed', log))}
            options.append(new_log)

    return options


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
            html.P(id='placeholder'),
            dcc.RadioItems(
                id='radio-type',
                options=[
                    {'label': 'Live', 'value': 'live'},
                    {'label': 'Completed training', 'value': 'done'}
                ],
                value='live'
            ),

            dcc.Dropdown(
                id='dropdown-live-file',
                placeholder="Select live log file",
                options=get_log_files("live"),
                clearable=True,
                disabled=False,
                searchable=True,
                multi=False,
                ),

            dcc.Dropdown(
                id='dropdown-done-file',
                placeholder="Select complete log file",
                options=get_log_files("done"),
                clearable=True,
                disabled=True,
                searchable=True,
                multi=False,
                ),

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


def update_graph(graph_id, graph_title, col, run_log_json, checklist_smoothing_options, slider_smoothing):
    """
    Function that updates the Graphs with new data from the logs
    :param graph_id: ID for Dash callbacks
    :param graph_title: Title displayed on layout
    :param col: name of column index for the data we want to retrieve
    :param run_log_json: the json file containing the data
    :param checklist_smoothing_options: 'yes' or 'no'
    :param slider_smoothing: value between 0 and 1, at interval of 0.05
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
            margin=go.layout.Margin(l=50, r=50, b=50, t=50),
            yaxis={'title': graph_title}
        )

        # Get the data from the json file stored in the page
        run_log_df = pd.read_json(run_log_json, orient='split')
        # Get the values for the curve
        step = run_log_df['Batch']
        values = run_log_df[col]

        # Apply Smoothing if needed
        if 'yes' in checklist_smoothing_options:
            values = smooth(values, weight=slider_smoothing)

        # Draw the curves
        trace_values = go.Scatter(
            x=step,
            y=values,
            mode='lines',
            name='Training'
        )

        figure = go.Figure(
            data=[trace_values],
            layout=layout
        )

        return dcc.Graph(figure=figure, id=graph_id)

    return dcc.Graph(id=graph_id)


@app.callback(Output('interval-log-update', 'interval'),
              [Input('dropdown-interval-control', 'value')])
def update_interval_log_update(interval_rate):
    """
    Callback that changes the interval time between updates for the graph.
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
    Callback that gets the csv file.
    :param _: holder value to trigger callback
    :return: data in json format
    """
    try:
        run_log_df = pd.read_csv(LOGFILE, header=0,
                                 dtype={'Batch': np.float64, 'Reward': np.float64, 'Loss': np.float64})
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
    Callback that gets the last Batch number from the JSON data.
    :param run_log_json: JSON log data
    :return: last row of the JSON log data
    """
    if run_log_json:
        run_log_df = pd.read_json(run_log_json, orient='split')
        return html.H6(f"Step: {run_log_df['Batch'].iloc[-1]}", style={'margin-top': '3px'})


@app.callback(Output('div-reward-graph', 'children'),
              [Input('run-log-storage', 'children'),
               Input('checklist-smoothing-options-reward', 'value'),
               Input('slider-smoothing-reward', 'value')])
def update_reward_graph(run_log_json, checklist_smoothing_options, slider_smoothing):
    """
    Callback that updates the reward graph
    :param run_log_json: JSON object of the data
    :param checklist_smoothing_options: Boolean to smooth the function
    :param slider_smoothing: Weight of the smoothing function
    :return: Graph
    """
    graph = update_graph('accuracy-graph', 'Reward', 'Reward', run_log_json,
                         checklist_smoothing_options, slider_smoothing)
    return [graph]


@app.callback(Output('div-loss-graph', 'children'),
              [Input('run-log-storage', 'children'),
               Input('checklist-smoothing-options-loss', 'value'),
               Input('slider-smoothing-loss', 'value')])
def update_loss_graph(run_log_json, checklist_smoothing_options, slider_smoothing):
    """
    Callback that updates the loss graph
    :param run_log_json: JSON object of the data
    :param checklist_smoothing_options: Boolean to smooth the function
    :param slider_smoothing: Weight of the smoothing function
    :return: Graph
    """
    graph = update_graph('cross-loss-graph', 'Loss', 'Loss',
                         run_log_json, checklist_smoothing_options, slider_smoothing)
    return [graph]


@app.callback(Output('div-current-reward-value', 'children'),
              [Input('run-log-storage', 'children')])
def update_div_current_reward_value(run_log_json):
    """
    Callback that updates the current reward value to be last reward value the agent got
    :param run_log_json: JSON object of the data
    :return: Paragraph with the value
    """
    if run_log_json:
        run_log_df = pd.read_json(run_log_json, orient='split')
        return [
            html.P(
                "Current Reward:",
                style={
                    'font-weight': 'bold',
                    'margin-top': '15px',
                    'margin-bottom': '0px'
                }
            ),
            html.Div(f"Reward: {run_log_df['Reward'].iloc[-1]}"),
        ]


@app.callback(Output('div-current-loss-value', 'children'),
              [Input('run-log-storage', 'children')])
def update_div_current_loss_value(run_log_json):
    """
    Callback that updates the current loss value to be last loss value the agent got
    :param run_log_json: JSON object of the data
    :return: Paragraph with the value
    """
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
            html.Div(f"Loss: {run_log_df['Loss'].iloc[-1]}"),
        ]


@app.callback(Output('dropdown-interval-control', 'multi'),
              [Input('dropdown-done-file', 'value'),
              Input('dropdown-live-file', 'value'),
              Input('radio-type', 'value')])
def change_log_file(value_done, value_live, value_type):
    """
    Callback that updates the log file to graph based on the dropdown selection
    :param value: Dropdown selection
    :return: Return value with no meaning for the purpose of compilation
    """
    global LOGFILE
    if(value_type == "live" and value_live is not None):
        LOGFILE = value_live
    elif(value_type == "done" and value_done is not None):
        LOGFILE = value_done
    else:
        LOGFILE = ''
        return False

@app.callback(Output('dropdown-live-file', 'disabled'),
              [Input('radio-type', 'value')])
def select_log_live(value):
    """
    Callback that disables dropdown menues based on the radio button selection
    :param value: Radio button selection
    :return: boolean
    """
    global LOGFILE
    if (value == "live"):
        return False
    else:
        LOGFILE = ''
        return True


@app.callback(Output('dropdown-done-file', 'disabled'),
              [Input('radio-type', 'value')])
def select_log_done(value):
    """
    Callback that disables dropdown menues based on the radio button selection
    :param value: Radio button selection
    :return: boolean
    """
    global LOGFILE
    if (value == "done"):
        return False
    else:
        LOGFILE = ''
        return True

# Running the server
if __name__ == '__main__':
    app.run_server(debug=True)
