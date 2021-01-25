import json

import numpy as np
import pandas as pd

import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

df = pd.read_pickle('Data/umap.pkl')
df = df.reset_index()
df = df.sort_values('label')

colors_list = px.colors.qualitative.Plotly
colors = {i:c for i,c in enumerate(colors_list)}

fig = px.scatter(df, x='x', y='y', color='label',
                 custom_data=[df.index, df.label])

fig.update_traces(marker_size=4)
fig.update_layout(clickmode='event+select',
                  height=600,
                  width=600,
                  legend={'itemsizing':'constant','itemwidth':60})

num_options = [{'label':s, 'value':s} for s in range(10)]
num_options.insert(0, {'label':'All', 'value':-1})
app.layout = html.Div([
        html.Div(
           html.Label(['Pick a label', dcc.Dropdown(options=num_options,
                                                    id='num-dropdown')]), 
            style={'width':'25%'}),
         html.Div(
                dcc.Graph(
                id='basic-interactions',
                figure=fig
        ),className='', style={'width':'50%', 'float':'left'}),
])

@app.callback(
    Output('basic-interactions', 'figure'),
    Input('num-dropdown', 'value')
)
def pick_label(val_label):

    if val_label is None:
        _df = df
    elif val_label == -1:
        _df = df
    else:
        _df = df[df.label == val_label].copy()
     
    f1 = px.scatter(_df, x='x', y='y', color='label',
                 custom_data=[_df.index, _df.label])

    f1.update_traces(marker_size=4)
    if val_label is not None and val_label >= 0:
        f1.update_traces(marker_color=colors[val_label])

    f1.update_layout(clickmode='event+select',
                  height=600,
                  width=600,
                  legend={'itemsizing':'constant','itemwidth':60})
    return f1
    

if __name__ == '__main__':
     app.run_server(debug=True, port=8601)