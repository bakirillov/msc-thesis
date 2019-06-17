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
from Bio.SeqFeature import SeqFeature, FeatureLocation
from dash.dependencies import Input, State, Output#, Event


class Logic():
    
    def __init__(self, config):
        self.cuda_avail = torch.cuda.is_available()
        with open(config, "r") as ih:
            self.states = json.load(ih)
        self.set_state(self.states[self.states["default"]])
        self.report_config = self.states["report"]
        self.k = self.states["k_neighbors"]
        self.offtarget_batch_size = self.states["offtarget_batch_size"]
        
    def set_state(self, state):
        #loading models
        if self.cuda_avail:
            self.on_model = torch.load(state["on"])
            self.off_model = torch.load(state["off"])
        else:
            self.on_model = torch.load(state["on"], map_location="cpu")
            self.off_model = torch.load(state["off"], map_location="cpu")
        self.reg_type = state["reg_type"]
        self.minl = state["min"]
        self.maxl = state["max"]
        if state["reg_type"] == "catboost":
            self.on_reg = CatBoostRegressor(
                random_seed=42,
                loss_function="RMSE"
            )
            self.on_reg.load_model(state["on_reg"])
        elif state["reg_type"] == "sklearn":
            self.on_reg = joblib.load(state["on_reg"])
        elif state["reg_type"] == "pytorch":
            if self.cuda_avail:
                self.on_reg = torch.load(state["on_reg"])
            else:
                self.on_reg = torch.load(state["on_reg"], map_location="cpu")
        #loading umap
        self.umap = joblib.load(state["umap"])
        #loading setups
        self.guide_length = state["guide_length"]
        self.use_pam = state["use_pam"]
        self.pam_before = state["pam_before"]
        self.pam = state["PAM"]
        #loading background
        with open(state["background"], "rb") as ih:
            self.background = pkl.load(ih) #sequences,representations,coordinates,labels
            
    def __getitem__(self, key):
        return(self.states[key])
    
    @staticmethod
    def get_pam_before(x, s, plen, length):
        return(x.extract(s)[0:plen])
    
    @staticmethod
    def get_pam_after(x, s, plen, length):
        return(x.extract(s)[length:length+plen])
    
    @staticmethod
    def getCandidate(start, end):
        l = FeatureLocation(start, end)
        f = SeqFeature(l, strand=1, type="sgRNA")
        return(f)
        
    def predict(self, X):
        return(
            self.on_model.batch_predict(X, 64, True)
        )
    
    def transform(self, X):
        return(
            self.umap.transform(X)
        )
    
    def regress(self, X):
        return(
            self.on_reg.predict(X)
        )
    
    def offtarget_predict(self, A, B):
        return(
            self.off_model.predict(A, B)
        )
    
    @staticmethod
    def count_mismatches(guide, target):
        return(np.sum([1 if a != b else 0 for a,b in zip(guide, target)]))
    
    @staticmethod
    def get_off_data(guides, targets):
        r = []
        ii = []
        for i,a in enumerate(guides):
            for b in tqdm(targets):
                ii.append(i)
                r.append(np.abs(a-b))
        return(np.array(r), np.array(ii))
    
    @staticmethod
    def compute_off_number(predictions, indices):
        i, c = np.unique(indices, return_counts=True)
        r = np.array([np.where(predictions[indices == a] == 1)[0].shape[0] for a in i])
        return(-r/np.max(r), r)
    
    @staticmethod
    def find_guides(s, pam_regex, length, before=True):
        plen = len(pam_regex.replace("[ATGC]", "N"))
        get_pam = Logic.get_pam_before if before else Logic.get_pam_after
        get_candidates = Logic.getCandidate(0, len(s))
        candidates = []
        for a in tqdm(np.arange(len(s))):
            candidate = (int(a), Logic.getCandidate(int(a), int(a)+length+plen))
            if re.search(pam_regex, get_pam(candidate[1], s, plen, length)):
                candidates.append(candidate)
        cut_pam = lambda x: x[plen:] if before else x[:-plen]
        guides = [(cut_pam(a[1].extract(s)), a[0], a[0]+length-1, get_pam(a[1], s, plen, length)) for a in candidates]
        return(guides)
    
    @staticmethod
    def fasta2line(s):
        return(
            "".join(list(filter(lambda x: not re.match("^>.+$", x), s.split("\n")))).replace(" ", "")
        )


class Reporter():
    
    def __init__(self, config):
        with open(config, "r") as ih:
            self.state = json.load(ih)
        self.env = Environment(loader=FileSystemLoader('.'))
        self.on = self.env.get_template(self.state["ontarget"])
        self.db = Vedis(":mem:")
        
    def get_on_html(self, all_data, cart):
        template_vars = {
            "gc": cart.to_html(),
            "ott": all_data.to_html()
        }
        return(self.on.render(template_vars))
    
    def get_pdf(self, html):
        randpath = Reporter.get_random_fn("pdf")
        HTML(string=html).write_pdf(
            randpath, stylesheets=[CSS(string='body { font-family: monospace !important }')]
        )
        return(randpath)
    
    @staticmethod
    def get_random_fn(ext):
        randpath = np.random.choice(np.arange(0, 10), (16,))
        randpath = "".join([str(a) for a in randpath])
        randpath = op.join("reports", randpath+"."+ext)
        return(randpath)
    
    @staticmethod
    def parse_vedis(s):
        ff = lambda x: x != ""
        u = str(s)
        for a in ["\\n", "b'[", "b\"[", "]'", "]\""]:
            u = u.replace(a, "")
        return(list(filter(ff, u.split(" "))))
    
    @staticmethod
    def val_float(x):
        if x[-1] == ".":
            x += "0"
        return(float(x))
    
    @staticmethod
    def val_int(x):
        return(int(re.sub("[^0-9\-]", "", x)))
    
    def get_data(self, sid):
        activities = [Reporter.val_float(a) for a in Reporter.parse_vedis(self.db[sid+"_activity"])]
        n_off = [Reporter.val_float(a) for a in Reporter.parse_vedis(self.db[sid+"_OTS"])]
        labels = [Reporter.val_int(a) for a in Reporter.parse_vedis(self.db[sid+"_label"])]
        strands = [Reporter.val_int(a) for a in Reporter.parse_vedis(self.db[sid+"_strand"])]
        gd = np.array(Reporter.parse_vedis(self.db[sid+"_guide"])).reshape((len(strands),4))
        act_n = [Reporter.val_int(a) for a in Reporter.parse_vedis(self.db[sid+"_#offtargets"])]
        return(activities, n_off, labels, strands, gd, act_n)
    
    def get_df(self, sid):
        activities, _, labels, strands, gd, n_off = self.get_data(sid)
        guides = [re.sub("[^ATGC]", "", a) for a in gd[:, 0]]
        starts = [Reporter.val_int(a) for a in gd[:, 1]]
        ends = [Reporter.val_int(a) for a in gd[:, 2]]
        pams = [re.sub("[^ATGC]", "", a) for a in gd[:, 3]]
        r = pd.DataFrame(
            {
               "guide": guides, 
               "label": labels,
               "activity": activities, "strand": strands,
               "start": starts, "end": ends, "PAM": pams, "#offtargets": n_off,
            }, columns=["guide", "label", "activity", "#offtargets", "strand", "start", "end", "PAM"]
        )
        return(r)
    
    def get_fasta(self, df):
        records = []
        for a in df.index:
            d = "activity:"+str(df.ix[a]["activity"])+";label:"+str(df.ix[a]["label"])
            d += ";strand:"+str(df.ix[a]["strand"])+";position:"+str(df.ix[a]["start"])+"-"
            d += str(df.ix[a]["end"])+";PAM:"+df.ix[a]["PAM"]+";#offtargets:"+str(df.ix[a]["#offtargets"])
            records.append(
                SeqRecord(
                    Seq(df.ix[a]["guide"], IUPAC.unambiguous_dna),
                    id=str(a), description=d
                )
            )
        randpath = Reporter.get_random_fn("fasta")
        SeqIO.write(records, randpath, "fasta")
        return(randpath)
    
    def get_csv(self, df):
        randpath = Reporter.get_random_fn("csv")
        df.to_csv(randpath)
        return(randpath)
    
    @staticmethod
    def zip_file(files):
        randpath = Reporter.get_random_fn("zip")
        with ZipFile(randpath, "w") as z: 
            for i in files:
                z.write(i)
                os.remove(i)
        return(randpath)

    
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

    
external_stylesheets = [
    "https://codepen.io/chriddyp/pen/bWLwgP.css", "https://codepen.io/chriddyp/pen/brPBPO.css"
]

logoworker = LogoWorker("./logos/")
image_directory = "./logos/"
static_image_route = "/static/"

if op.exists(image_directory):
    shutil.rmtree(image_directory)
    os.makedirs(image_directory)

server = flask.Flask(__name__)
app = dash.Dash(server=server)#, external_stylesheets=external_stylesheets)
app.title = "CRISPR CAD"
session_id = str(uuid.uuid4())
app.layout = html.Div([
    html.Div(session_id, id='session-id', style={'display': 'none'}),
    dcc.Tabs(id="tabs", value="tab-1", children=[
        dcc.Tab(label="Settings", value="tab-1"),
        dcc.Tab(label="Operations", value="tab-2")
    ]),
    html.Div(id="tabs-content"),
    dcc.Store(id="effector-store"),
    dcc.Store(id="dna-store"),
    dcc.Store(id="cart-store"),
    dcc.Store(id="mode-store"),
    dcc.Store(id="organism-store"),
    dcc.Store(id="ontarget-report-store"),
    dcc.Store(id="mismatch-store")
])
app.config['suppress_callback_exceptions']=True
logic = Logic("config.json")
reporter = Reporter("report.config.json")
effectors = np.array([a for a in logic.states])
effectors = effectors[effectors != "report"]
effectors = effectors[effectors != "default"]
effectors = effectors[effectors != "off_target_indices"]
organisms = np.array([a for a in logic.states["off_target_indices"]])
dropdown_options = [{"label": a, "value": a} for a in effectors]
organisms_options = [{"label": a, "value": a} for a in organisms]
plotter = Plotter()
with open(logic.states["off_target_indices"][0], "rb") as ih:
    current_index = pkl.load(ih)
app_state = {}
plausible_ots = {}
offtarget_predictions = {}


@app.callback(
    Output("tabs-content", "children"),
    [Input("tabs", "value"), Input("session-id", "children")],
    [
        State("dna-store", "data"), State("effector-store", "data"),
        State("organism-store", "data"), State("cart-store", "data"),
        State("mode-store", "data"), State("mismatch-store", "data")
    ]
)
def render_content(tab, sid, dna, effector, organism, cart, mode, n_mm):
    if not n_mm:
        n_mm = 6
    else:
        n_mm = n_mm["mismatches"]
    if not mode:
        mode = "space"
    else:
        mode = mode["mode"]
    if not cart:
        cart = []
    else:
        cart = cart["cart"]
    if effector:
        effector = effector["effector"]
    if tab == "tab-1":
        return(
            html.Div([input_sector(), copyright()])
        )
    elif tab == "tab-2":
        d = None
        if dna and dna["dna"] != "" and effector:
            dna = dna["dna"]
            input_data = Logic.fasta2line(dna)
            guides = Logic.find_guides(
                input_data, logic.pam, logic.guide_length, logic.pam_before
            )
            guides = list(filter(lambda x: len(x[0]) == logic.guide_length, guides))
            strands = [1]*len(guides)
            guides.extend(
                Logic.find_guides(
                    "".join(list(reversed(input_data))), logic.pam, 
                    logic.guide_length, logic.pam_before
                )
            )
            guides = list(filter(lambda x: len(x[0]) == logic.guide_length, guides))
            strands.extend(
                [-1]*(len(guides)-len(strands))
            )
            guides_oh = np.stack([correct_order(onehot(a[0])) for a in guides])
            internal, _, length, label, _ = logic.predict(guides_oh)
            internal = internal.reshape((internal.shape[0], 64))
            coords = logic.transform(internal)
            activity = logic.regress(internal)
            app_state["guides"] = guides
            app_state["sequences"] = np.array([a[0] for a in guides])
            app_state["representations"] = internal
            app_state["coords"] = coords
            app_state["activities"] = activity
            app_state["labels"] = label
            app_state["strands"] = strands
            #plausible_ots = []
            print("Searching for plausible off-targets")
            full_targets = np.array([
                b+p[0] if not logic.pam_before else p[-1]+b 
                    for b,p in list(zip(current_index["sequences"], current_index["PAM"]))
            ])
            full_targets_oh = np.stack([correct_order(onehot(a)) for a in full_targets])
            full_guides = [a[0]+a[-1][0] if not logic.pam_before else a[-1][-1]+a[0] for a in guides]
            for full_guide in tqdm(full_guides):
                full_guide_oh = correct_order(onehot(full_guide))
                plausible_ots[full_guide] = []
                offtarget_predictions[full_guide] = []
                for b in np.arange(0, len(full_targets), logic.offtarget_batch_size):
                    current_batch = full_targets_oh[b:b+logic.offtarget_batch_size]
                    current_mms = np.apply_along_axis(lambda x: np.sum(np.abs(x-full_guide_oh)), 1, current_batch)
                    current_targets = full_targets[b:b+logic.offtarget_batch_size]
                    current_targets_oh = current_batch[current_mms<=n_mm*4]
                    current_guide_oh = np.tile(
                        full_guide_oh, current_targets_oh.shape[0]
                    ).reshape(current_targets_oh.shape[0], current_targets_oh.shape[1])
                    current_predictions = np.argmax(
                        logic.offtarget_predict(current_guide_oh, current_targets_oh), 1
                    )
                    final_ots = current_targets[current_mms<=n_mm*4][current_predictions == 1]
                    plausible_ots[full_guide].extend(list(final_ots))
            act_n = np.array([len(plausible_ots[a]) if a in plausible_ots else 0 for a in full_guides])
            n_off = -act_n/np.max(act_n)
            app_state["act_n"] = act_n
            app_state["n_off"] = n_off
            d = {
                "coords": coords, "activity": activity, "guide": guides, 
                "label": label, "strand": strands, "OTS": n_off, "#offtargets": act_n
            }
            for a in d:
                if a != "coords":
                    reporter.db[sid+"_"+a] = d[a]
                else:
                    reporter.db[sid+"_UMAP1"] = d[a][:,0]
                    reporter.db[sid+"_UMAP2"] = d[a][:,1]
        return(
            html.Div(
                [
                    work_sector(d, effector, cart, mode), copyright()
                ]
            )
        )
    
def about_sector():
    return(
        "RESERVED AREA FOR ABOUT PAGE  "+"""Lorem ipsum dolor sit amet, consectetur adipiscing elit. Etiam maximus ornare ultricies. Sed at magna vitae magna vulputate varius. Suspendisse vitae diam vitae nulla bibendum pretium id eu enim. Curabitur sed sem quis ante mollis feugiat. Nullam facilisis nunc sed massa venenatis, vel pulvinar libero faucibus. Ut libero risus, sodales vitae lectus vel, interdum luctus neque. Nullam leo massa, tempus et euismod eget, vehicula sit amet neque. Aliquam mi lorem, pulvinar eu lectus vel, dignissim viverra ex. Fusce mattis lacus et erat sollicitudin, et euismod nulla mollis. Nunc elit nisl, viverra non nunc vel, maximus posuere nibh. Nullam eu sapien egestas, auctor enim ac, venenatis tellus. Curabitur facilisis sit amet odio viverra iaculis. Nulla ullamcorper sagittis magna. Curabitur varius turpis ut leo porttitor dignissim. Nam maximus sit amet augue vitae pharetra.Nunc at lobortis magna. Aliquam erat volutpat. Curabitur posuere eu ligula at gravida. """
    )

def input_sector():
    return(
        html.Div([
            html.Center([html.H3(children='CRISPR CAD: interpretable ML-based search for CRISPR/Cas9 and Cpf1 guides')]),
            html.Div([
                dcc.Textarea(
                    placeholder='Paste gene of interest here in fasta format...',
                    value='',
                    style={'width': '100%', "height": "700px"},
                    id="input-dna"
                ), about_sector()
            ], className="nine columns"),
            html.Div([
                dcc.Upload(
                    id='upload-data',
                        children=html.Div([
                        'Drag and drop or ',
                        html.A('Select fasta file')
                    ]),
                    style={
                        'height': '90px',
                        'borderStyle': 'dashed'
                    },
                    multiple=False
                ),
                html.Hr(),
                "Choose effector:",
                dcc.Dropdown(
                    options=dropdown_options,
                    value=logic.states["default"], id="effector-choice"
                ),
                html.Hr(),
                "Choose organism:",
                dcc.Dropdown(
                    options=organisms_options,
                    value=organisms[0], id="organism-choice"
                ),
                html.Hr(),
                dcc.Slider(
                    id='mismatch-slider',
                    min=1,
                    max=21,
                    step=1,
                    value=6,
                ),
                "Allowed mismatches in off-target search",
                html.Hr(),
                authors_and_paper()
            ], className="two columns")
            ], 
            className="row"
        )
    )

@app.callback(
    Output("mismatch-store", "data"),
    [Input('mismatch-slider', 'value')])
def update_mismatch_store(value):
    return({"mismatches": value})

def authors_and_paper():
    return(
        html.Center(
            "RESERVED AREA FOR INFO ABOUT AUTHORS AND LINK TO THE PAPER"
        )
    )

def copyright():
    return(
        html.Center(
            "RESERVED AREA FOR LICENSE AND COPYRIGHT INFO"
        )
    )

def work_sector(guides, effector, cart, mode):
    xaxis = ""
    yaxis = ""
    if guides:
        info = []
        for a,b,c,d,e in zip(guides["guide"], guides["activity"], guides["label"], guides["strand"], guides["#offtargets"]):
            info.append(
                str(a)+"; Activity:"+str(b)+"; Label:"+str(c)+"; Strand:"+str(d)+"; "+"#offtargets:"+str(e)+";")
        if mode == "pareto":
            traces = [
                plotter.get_offtargets(
                    guides["activity"], guides["OTS"], info
                )   
            ]
            xaxis = "On-target activity"
            yaxis = "Negative max-normalized number of off-targets"
        elif mode == "space":
            traces = [
                plotter.get_background(
                    np.array(logic.background["coordinates"]), 
                    np.array(logic.background["labels"]), 
                    np.array(logic.background["labels"])
                ),
                plotter.get_guide(guides["coords"], guides["label"], guides["activity"], info)
            ]
            xaxis = "UMAP 1"
            yaxis = "UMAP 2"
    else:
        traces = [plotter.empty_plot()]
    return(
        html.Div([
            html.Div(
                [
                    dcc.Dropdown(
                        options=[
                            {"label": "Guide space", "value": "space"},
                            {"label": "Pareto front", "value": "pareto"}
                        ], value="space", id="mode-choice"
                    )
                ]
            ),
            html.Div(
                [
                    html.Div([dcc.Graph(
                        id='g2', 
                        figure=plotter.make_plot(traces, xaxis=xaxis, yaxis=yaxis)
                    )], id="plot_space", className="ten columns"),
                    html.Div(
                        [
                            html.Center(html.H4("Guide cart")),
                            html.Div([guidecart(cart)], id="guide-cart", style={'fontSize': '12px'}),
                            html.Hr(),
                            html.Center([html.Button("Report", id="repbutton")]),
                            html.Div([], id="report-link"),
                            html.Hr(),
                            html.Img(id='ontargetvicinity'),
                            html.Br(),
                            "On-target vicinity",
                            html.Hr(),
                            html.Img(id="offtargetmm"),
                            html.Br(),
                            "Off-target mismatch composition"
                        ], className="two columns")
                ], className="row"
            )
        ])
    )

def guidecart(guides):
    if len(guides) > 0:
        if len(guides) < 20:
            shown = [a for a in guides] 
        else: 
            shown = [a for a in guides[0:19]]
            shown.append("+ "+str(len(guides)-19)+" guides")
    else:
        shown = ["Cart is empty"]
    return(
        html.Ul([html.Li(a) for a in shown])
    )

@app.callback(
    Output("cart-store", "data"), [Input("g2", "clickData")], [State("cart-store", "data")]
)
def change_guide_cart(clickData, cart):
    cD = list(filter(lambda x: "text" in x, clickData["points"]))[0]
    sequence = re.search("[ATGC]+", cD["text"])[0]
    current = sequence.replace(" ", "")
    selected = [current]
    if cart:
        if "in cart" in cD["text"] or current in cart["cart"]:
            ck = np.array(cart["cart"])
            selected = list(ck[ck != current])
        else:
            selected = cart["cart"]+selected
    return({"cart": selected})

@app.callback(
    Output("guide-cart", "children"),
    [Input("cart-store", "data")], [State("guide-cart", "children")]
)
def show_guide_cart(cart, current_cart):
    if cart:
        cart = cart["cart"]
    else:
        cart = []
    return(
        guidecart(cart)
    )

@app.callback(
    Output("ontargetvicinity", "src"),
    [Input("cart-store", "data")]#, Input('session-id', 'children'), Input("mode-store", "data")]
)
def update_ontarget_logo(cart):
    if cart:
        current_sequence = [a for a in cart["cart"]][-1]
        current_onehot = correct_order(onehot(current_sequence))
        current_representation = logic.predict(
            current_onehot.reshape(1, len(current_sequence)*4)
        )[0].reshape(1,64)
        current_coords = logic.transform(current_representation).reshape(2)
        current_d2b = np.apply_along_axis(lambda a: np.sum(np.abs(a-current_coords)), 1, logic.background["coordinates"])
        top_k = list(sorted(np.arange(current_d2b.shape[0]), key=lambda x: current_d2b[x]))
        top_k = np.array(top_k[0:logic.k])
        top_k_seq = np.array(logic.background["sequences"])[top_k]
        img_fn = "".join([str(a) for a in np.random.choice(np.arange(10), 12)])+".png"
        logoworker.logo_of_list(top_k_seq, img_fn)
        return(static_image_route+img_fn)

@app.callback(
    Output("offtargetmm", "src"),
    [Input("cart-store", "data")]
)
def update_offtarget_logo(cart):
    if cart:
        current_sequence = [a for a in cart["cart"]][-1]
        current_guide = list(filter(lambda x: x[0] == current_sequence, app_state["guides"]))[0]
        if not logic.pam_before:
            current_full_guide = current_guide[0]+current_guide[-1][0]
        else:
            current_full_guide = current_guide[-1][-1]+current_guide[0]
        img_fn = "".join([str(a) for a in np.random.choice(np.arange(10), 12)])+".png"
        logoworker.logo_of_list(plausible_ots[current_full_guide], img_fn)
        return(static_image_route+img_fn)
        
@app.callback(
    Output("plot_space", "children"),
    [Input("cart-store", "data"), Input('session-id', 'children'), Input("mode-store", "data")], 
    [State("plot_space", "children"), State("effector-store", "data")]
)
def plot_cart(cart, sid, mode, old_plot, effector):
    if not mode:
        mode = "space"
    else:
        mode = mode["mode"]
    if effector:
        effector = effector["effector"]
    activity = app_state["activities"]
    n_off = app_state["n_off"]
    labels = app_state["labels"]
    guides = app_state["guides"]
    strands = app_state["strands"]
    act_n = app_state["act_n"]
    info = []
    in_cart = np.array([a for a in cart["cart"]]) if cart else np.array([])
    for a,b,c,d,e in zip(guides, activity, labels, strands, act_n):
        gs = ",".join([a[0], str(a[1]), str(a[2]), a[3]])
        info.append(
            gs+"; Activity:"+str(b)+"; Label:"+str(c)+"; Strand"+str(d)+"; Off-targets:"+str(e)
        )
    gc = np.array([a.split(",")[0] in in_cart for a in info])
    xaxis = ""
    yaxis = ""
    if mode == "pareto":
        traces = [
            plotter.get_offtargets(
                activity[np.invert(gc)], n_off[np.invert(gc)], np.array(info)[np.invert(gc)]
            ),
            plotter.get_pareto_cart(
                activity[gc], n_off[gc], [a+" in cart;" for a in np.array(info)[gc]]
            )
        ]
        xaxis = "On-target activity"
        yaxis = "Negative max-normalized number of off-targets"
    elif mode == "space":
        coords = app_state["coords"]
        traces = [
            plotter.get_background(
                np.array(logic.background["coordinates"]), 
                np.array(logic.background["labels"]), 
                np.array(logic.background["labels"])
            ),
            plotter.get_guide(
                coords[np.invert(gc)], labels[np.invert(gc)], 
                activity[np.invert(gc)], np.array(info)[np.invert(gc)]
            ),
            plotter.get_guide_cart(
                coords[gc], labels[gc], 
                activity[gc], np.array(info)[gc]
            )
        ]
        xaxis = "UMAP 1"
        yaxis = "UMAP 2"
    return(
        [
            dcc.Graph(
                id='g2', 
                figure=plotter.make_plot(traces, xaxis=xaxis, yaxis=yaxis)
            )
        ]
    )

@app.callback(Output("effector-store", "data"),
              [Input("effector-choice", "value")])
def set_effector(effector):
    return(
        {"effector": effector}
    )

@app.callback(Output("organism-store", "data"),
              [Input("organism-choice", "value")])
def set_organism(organism):
    return(
        {"organism": organism}
    )

@app.callback(Output("dna-store", "data"),
              [Input("input-dna", "value")])
def set_dna(dna):
    return(
        {"dna": dna}
    )

@app.callback(Output("mode-store", "data"),
              [Input("mode-choice", "value")])
def set_mode(mode):
    return(
        {"mode": mode}
    )

@server.route("/reports/<path:path>")
def download(path):
    """Serve a file from the upload directory."""
    return(flask.send_from_directory("reports", path, as_attachment=True))

def file_download_link(filename):
    """Create a Plotly Dash 'A' element that downloads a file from the app."""
    location = "{}".format(urlquote(filename))
    return(html.A(filename, href=location))

@app.server.route('{}<image_path>.png'.format(static_image_route))
def serve_image(image_path):
    image_name = '{}.png'.format(image_path)
    return(flask.send_from_directory(image_directory, image_name))

@app.callback(
    Output("report-link", "children"),
    [Input("repbutton", "n_clicks")], 
    [State("session-id", "children"), State("cart-store", "data")]
)
def report(n, sid, cart):
    if n:
        try:
            df = reporter.get_df(sid)
        except:
            return(html.Center("No report is produced"))
        else:
            fns = []
            if cart:
                #cart_n = [int(a.split(":")[0]) for a in cart["cart"]]
                b = df["guide"].apply(lambda x: x in cart["cart"])
                cart_df = df[b]
                fns.append(reporter.get_fasta(cart_df))
            else:
                cart_df = pd.DataFrame({"status":["none chosen"]})
            df_s = df.sort_values(by="activity", ascending=False)
            html_s = reporter.get_on_html(df_s, cart_df)
            fns.append(reporter.get_pdf(html_s))
            fns.append(reporter.get_fasta(df_s))
            file = Reporter.zip_file(fns)
            link = file_download_link(file)
            return(html.Center(link))

@app.callback(Output('input-dna', 'value'),
    [Input('upload-data', 'contents')],
    [State('upload-data', 'filename')])
def load_file(contents, filename):
    if contents != None:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        name, extension = op.splitext(filename)
        try:
            if extension in [".fasta", ".fa", ".FASTA", ".FA"]:
                return(io.StringIO(decoded.decode('utf-8')).read())
            else:
                return("Only fasta files are supported")
        except Exception as e:
            return("There was an error processing this file: "+str(e))
        
        
if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=1488, debug=False)
