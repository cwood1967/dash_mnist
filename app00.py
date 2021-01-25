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

fig = px.scatter(df, x='x', y='y', color='label',
                 custom_data=[df.index, df.label])

fig.update_traces(marker_size=4)
fig.update_layout(clickmode='event+select',
                  height=600,
                  width=600,
                  legend={'itemsizing':'constant','itemwidth':60})

app.layout = html.Div([
         html.Div(
                dcc.Graph(
                id='basic-interactions',
                figure=fig
        ),className='', style={'width':'50%', 'float':'left'}),
])
    
if __name__ == '__main__':
     app.run_server(debug=True, port=8600)