import re
import io
import os
import dash
import time
import json
import uuid
import flask
import base64
import shutil
from reviewed_core import *
import numpy as np
import pandas as pd
import pickle as pkl
import os.path as op
from Bio import SeqIO
from tqdm import tqdm
from vicinity import *
from vedis import Vedis
from Bio.Seq import Seq
#from logic import Logic
from zipfile import ZipFile
from Bio.Alphabet import IUPAC
import plotly.graph_objs as go
from operator import itemgetter
from weasyprint import HTML, CSS
import dash_core_components as dcc
from Bio.SeqRecord import SeqRecord
import dash_html_components as html
from sklearn.externals import joblib
from catboost import CatBoostRegressor
from urllib.parse import quote as urlquote
from scipy.spatial.distance import euclidean
from jinja2 import Environment, FileSystemLoader
from dash.dependencies import Input, State, Output

class Plotter():

    def __init__(self):
        self.label_colors = {
            0:"blue", 1:"red", 2:"black"
        }
        #self.hovermode = "closest"
        self.margin = {'l': 40, 'b': 40, 't': 10, 'r': 10}
        self.legend={'x': 0, 'y': 1}
        self.sizes = [16, 32, 8]
        self.line_width = 2
        
        
    def get_background(self, current_data, current_labels, current_clusters):
        return(
            go.Scatter(#3d(
                x=current_data[:,0],
                y=current_data[:,1],
                #z=current_data[:,2],
                mode="markers",
                hoverinfo="none",
                marker=dict(
                    size=self.sizes[2],
                    color = current_clusters, #set color equal to a variable
                    colorscale='Blues',
                    showscale=True,
                    line = dict(
                        color = current_labels,#[self.label_colors[a] for a in current_labels],
                        width = self.line_width
                    )
                )
            )
        )
            
    def get_guide(self, current_data, current_labels, current_clusters, current_info):
        return(
            go.Scatter(
                x=current_data[:,0],
                y=current_data[:,1],
                #z=current_data[:,2],
                mode="markers",
                text=[str(i)+": "+current_info[i] for i in np.arange(current_data.shape[0])],
                hoverinfo="text",
                #textposition='top center',
                textfont=dict(
                family='sans serif',
                    size=16,
                    color="darkgreen"
                ),
                marker=dict(
                    size=self.sizes[1],
                    color = current_clusters, #set color equal to a variable
                    colorscale='Reds',
                    showscale=False,
                    line = dict(
                        color = current_labels,#[self.label_colors[a] for a in current_labels],
                        width = self.line_width
                    )
                )
            )
        )
    
    def get_guide_cart(self, coords, labels, clusters, info):
        return(
            go.Scatter(
                x=coords[:,0],
                y=coords[:,1],
                mode="markers",
                text=[str(i)+": "+info[i] for i in np.arange(clusters.shape[0])],
                hoverinfo="text",
                marker=dict(
                    size=self.sizes[1],
                    showscale=False,
                    color="yellow",
                    line = dict(
                        color = "red",
                        width = self.line_width
                    )
                )
            )
        )
    
    def get_offtargets(self, activity, off_n, current_info):
        return(
            go.Scatter(
                x=activity,
                y=off_n,
                mode="markers",
                text=[str(i)+": "+current_info[i] for i in np.arange(activity.shape[0])],
                hoverinfo="text",
                marker=dict(
                    size=self.sizes[0],
                    #color="yellow",
                    showscale=False
                )
            )
        )
    
    def get_pareto_cart(self, activity, off_n, current_info):
        return(
            go.Scatter(
                x=activity,
                y=off_n,
                mode="markers",
                text=[str(i)+": "+current_info[i] for i in np.arange(activity.shape[0])],
                hoverinfo="text",
                marker=dict(
                    size=self.sizes[0],
                    showscale=False,
                    color="yellow",
                    line = dict(
                        color = "red",
                        width = self.line_width
                    )
                )
            )
        )
    
    def empty_plot(self):
        trace0 = go.Scatter(
            x=[],
            y=[]
        )
        trace1 = go.Scatter(
            x=[],
            y=[]
        )
        data = [trace0, trace1]
        return(data)
        
    def make_plot(self, traces, xaxis="UMAP1", yaxis="UMAP2"):
        return({
            'data': traces,
            'layout': go.Layout(
                xaxis={'title': xaxis},
                yaxis={'title': yaxis},
                margin=self.margin,
                height=900,
                legend=self.legend
            )
        })
    
    def empty_df(self):
        return(
            pd.DataFrame(
                {
                    "#":[], "guide": [], "class": [], "activity": [], 
                    "mean distance": [], "position": [], "strand": [], "cluster": []
                }, 
                columns=[
                    "#", "guide", "class", "activity", 
                    "mean distance", "position", "strand", "cluster"
                ]
            )
        )

plotter = Plotter()

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "My own private findface"
app.layout = html.Div(children=[
    html.H1(children='My own private findface'),
    dcc.Graph(id="g2", figure=)
])

#app.config['suppress_callback_exceptions']=True

app.run_server(debug=False)