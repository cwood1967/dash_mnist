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

def graphimage(data, point=0):
    try:
        number = data['points'][point]['customdata'][0]
        img = 255*images[number]
        img = img.astype(np.uint8)
    except:
        img = np.zeros((28,28))

    z = np.zeros((28,28,3), dtype=np.uint8)
    z[:,:,1] =  img
    f = go.Figure(go.Image(z=z))
    f.update_layout(
        height=96,
        width=96,
        margin=dict(l=4,r=4,t=4,b=4)
    )
    f.update_xaxes(showticklabels=False) 
    f.update_yaxes(showticklabels=False) 
    return f

def imagediv(fg, id):
    res = html.Div([
        dcc.Graph(id=id, figure=fg)],
        className='',
        #style={'width':'200px'}
    )
    return res
    
colors_list = px.colors.qualitative.Plotly
colors = {i:c for i,c in enumerate(colors_list)}
print(colors)
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

images = np.load('mnist_norm.npy')
images = images.reshape((-1, 28, 28))

df = pd.read_pickle('umap.pkl')#.sort_values('label')
df = df.reset_index()
df = df.sort_values('label')

styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll'
    }
}
fig = px.scatter(df, x='x', y='y', color='label',
                 custom_data=[df.index, df.label])

fig.update_traces(marker_size=4)
fig.update_layout(clickmode='event+select',
                  height=600,
                  width=600,
                  legend={'itemsizing':'constant','itemwidth':60})

z = np.zeros((28,28,3), dtype=np.uint8)
z[:,:,1] =255*(images[14]).astype('uint8') 
mnistfig = go.Figure(go.Image(z=z))
selfig = go.Figure() # make_subplots(1,4, horizontal_spacing=.05)
mnistfig.update_layout(height=282,
                       width=282)
mnistfig.update_xaxes(showticklabels=False) 
mnistfig.update_yaxes(showticklabels=False) 

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
        html.Div([
            html.Div(dcc.Graph(id='mnist-image',
                            figure=mnistfig), 
                className='',
                style={}),
            html.Div(
                children=[imagediv(selfig, "s0")],
                id='selected-images', className='', 
                style={'width':'95%', 'height':'300px', 'display':'flex',
                       'flex-wrap':'wrap'}),
            html.Div([
                dash_table.DataTable(
                    id='table',
                    columns=[{'name':i, 'id': i} for i in df.columns],
                    data=df.sample(10).to_dict('records'),
                    style_cell={'font-size':'medium', 'height':'10px'}
                ),     
            ], style={'position':'absolute', 'width':'100%','bottom':'0px', 'left':'0px',
                      }),
    ], style={'width':'45%', 'position':'absolute', 'height':'750px',
              'left':'50%', 'top':'50px','border':'3px solid #73AD21'}),
])

@app.callback(
    Output('basic-interactions', 'figure'),
    Input('num-dropdown', 'value')
)
def pick_label(val_label):
    '''
    callback for picking a single label to display. Input is from the dropdown,
    Output is a plotly express figure scatter plot.
    Parameters
    ---------
    val_label : int
        The integer number from the dropdown
        
    Returns
    -------
    f1 : Figure
        Scatter plot of the selected digit
    '''
    if val_label is None:
        _df = df
        xcolor = 'label'
    elif val_label == -1:
        _df = df
        xcolor = 'label'
    else:
        _df = df[df.label == val_label].copy()
        xcolor = 'xcolor'
        _df['xcolor'] = colors[val_label]
        print(xcolor, val_label, colors[val_label])
        print(_df.head())
     
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
    
@app.callback(
    Output('mnist-image', 'figure'),
    Input('basic-interactions', 'hoverData'))
def display_hover_data(hoverData):
    '''
    Callback to display the digit under the mouse hover
    '''
    f = graphimage(hoverData)
    return f 

if __name__ == '__main__':
     app.run_server(debug=True)