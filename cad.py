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
from time import time
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
from Bio.SeqFeature import SeqFeature, FeatureLocation


class Logic():
    
    def __init__(self, config):
        self.cuda_avail = torch.cuda.is_available()
        with open(config, "r") as ih:
            self.states = json.load(ih)
        self.set_state(self.states[self.states["default"]])
        self.on_report = self.states["on-target report"]
        self.off_report = self.states["on-target report"]
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
        for m in self.on_model.modules():
            if 'Conv' in str(type(m)):
                setattr(m, 'padding_mode', 'zeros')
        for m in self.off_model.modules():
            if 'Conv' in str(type(m)):
                setattr(m, 'padding_mode', 'zeros')
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
        D = A-B
        return(
            self.off_model.batch_predict(D, 128, True)[2]
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

    
def get_random_fn(ext):
    randpath = np.random.choice(np.arange(0, 10), (16,))
    randpath = "".join([str(a) for a in randpath])
    randpath = op.join("reports", randpath+"."+ext)
    return(randpath)

def zip_file(files):
    randpath = get_random_fn("zip")
    with ZipFile(randpath, "w") as z: 
        for i in files:
            z.write(i)
            os.remove(i)
    return(randpath)

def get_fasta(df):
    ds = []
    for a in df.index:
        d = ""
        for b in df.columns:
            #u = df.loc[a][b] if type(df.loc[a][b]) != tuple else ";".join([str(c) for c in df.loc[a][b]])
            d += b+":"+str(df.loc[a][b])+";"
        ds.append(d)
    records = [
        SeqRecord(Seq(a, IUPAC.unambiguous_dna), description=d) 
            for a,b in zip(df["sequence"].values, ds)
    ]
    randpath = get_random_fn("fasta")
    SeqIO.write(records, randpath, "fasta")
    return(randpath)

def get_on_html(data, template_file):
    env = Environment(loader=FileSystemLoader('.'))
    on = env.get_template(template_file)#config["on-target report"])
    templates_vars = {
        "gc": data.to_html()
    }
    chosen = data[data["chosen?"] == True]
    templates_vars["ott"] = chosen.to_html()
    html = on.render(templates_vars)
    randpath_pdf = get_random_fn("pdf")
    HTML(string=html).write_pdf(
        randpath_pdf, stylesheets=[CSS(string='body { font-family: monospace !important }')]
    )
    randpath_fasta_all = get_fasta(data)
    randpath_fasta_chosen = get_fasta(chosen)
    return(randpath_pdf, randpath_fasta_all, randpath_fasta_chosen)

def get_off_html(data, template_file):
    env = Environment(loader=FileSystemLoader('.'))
    off = env.get_template(template_file)
    templates_vars = {
        "gc": data.to_html()
    }
    html = off.render(templates_vars)
    randpath_pdf = get_random_fn("pdf")
    HTML(string=html).write_pdf(
        randpath_pdf, stylesheets=[CSS(string='body { font-family: monospace !important }')]
    )
    randpath_fasta = get_fasta(data)
    return(randpath_pdf, randpath_fasta)

    
class Plotter():

    def __init__(self):
        self.label_colors = {
            0:"blue", 1:"red", 2:"black"
        }
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

PORT = int(os.environ.get('PORT', 5000))


if not op.exists("./logos/"):
    os.makedirs("./logos/")
else:
    shutil.rmtree("./logos/")
    os.makedirs("./logos/")

if not op.exists("./reports/"):
    os.makedirs("./reports/")
else:
    shutil.rmtree("./reports/")
    os.makedirs("./reports/")

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
        dcc.Tab(label="On-target", value="tab-2"),
        dcc.Tab(label="Off-target", value="tab-3")
    ]),
    html.Div(id="tabs-content"),
    dcc.Store(id="effector-store"),
    dcc.Store(id="dna-store"),
    dcc.Store(id="cart-store"),
    dcc.Store(id="organism-store"),
    dcc.Store(id="ontarget-report-store"),
    dcc.Store(id="mismatch-store")
])
app.config['suppress_callback_exceptions']=True
logic = Logic("config.json")
#reporter = Reporter(logic.states["report"])#"report.config.json")
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
offtarget_predictions = {}


@app.callback(
    Output("tabs-content", "children"),
    [Input("tabs", "value"), Input("session-id", "children")],
    [
        State("dna-store", "data"), State("effector-store", "data"),
        State("organism-store", "data"), State("cart-store", "data"),
        State("mismatch-store", "data")
    ]
)
def render_content(tab, sid, dna, effector, organism, cart, n_mm):
    #print(sid in app_state)
    plausible_ots = {}
    if sid not in app_state:
        print("Not in state")
        app_state[sid] = {}
        app_state[sid]["cart"] = []
        app_state[sid]["offtargets"] = {}
    if not n_mm:
        n_mm = 6
    else:
        n_mm = n_mm["mismatches"]
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
    elif tab == "tab-3":
        guides = [a.split(",")[0] for a in app_state[sid]["cart"]]
        guides_oh = np.stack([correct_order(onehot(a)) for a in guides])
        guides_a = logic.regress(logic.predict(guides_oh)[0].reshape((guides_oh.shape[0], 64)))
        app_state[sid]["cart_activities"] = guides_a
        pams = [a.split(",")[3][0] for a in app_state[sid]["cart"]]
        full_guides = [a+b if not logic.pam_before else b+a for a,b in zip(guides, pams)]
        full_targets = np.array([
            b+p[0] if not logic.pam_before else p[-1]+b 
                for b,p in list(zip(current_index["sequences"], current_index["PAM"]))
        ])
        full_targets_oh = np.stack([correct_order(onehot(a)) for a in full_targets])
        plausible_ots = {}
        mmsses = []
        ohs = []
        start_time = time()
        for full_guide in tqdm(full_guides):
            ohs.append(correct_order(onehot(full_guide)))
            mmsses.append(
                np.apply_along_axis(lambda x: np.sum(np.abs(x-ohs[-1])), 1, full_targets_oh)
            )
        print("all "+str(time()-start_time))
        for full_guide, full_guide_oh, current_mms in tqdm(zip(full_guides, ohs, mmsses)):
            plausible_ots[full_guide] = []
            offtarget_predictions[full_guide] = []
            start_time = time()
            current_targets_oh = full_targets_oh[current_mms<=n_mm*4]
            current_guide_oh = np.tile(
                    full_guide_oh, current_targets_oh.shape[0]
                ).reshape(current_targets_oh.shape[0], current_targets_oh.shape[1])
            current_predictions = np.argmax(
                logic.offtarget_predict(current_guide_oh, current_targets_oh), 1
            )
            plausible_ots[full_guide] = list(full_targets[current_mms<=n_mm*4][current_predictions == 1])
            print(time()-start_time)
        n_off = np.array([len(plausible_ots[a]) if a in plausible_ots else 0 for a in full_guides])
        act_n = -n_off/np.max(n_off)
        app_state[sid]["n_off"] = n_off
        app_state[sid]["norm_n_off"] = act_n 
        d = {"X": guides_a, "Y": act_n}
        app_state[sid]["offtargets"] = plausible_ots
        return(
            #"ololo"
            html.Div(
                [
                    offtarget_sector(d, effector, cart, sid), copyright()
                ]
            )
        )
    elif tab == "tab-2":
        print("tab-2", app_state[sid]["cart"])
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
            app_state[sid]["guides"] = guides
            app_state[sid]["sequences"] = np.array([a[0] for a in guides])
            app_state[sid]["representations"] = internal
            app_state[sid]["coords"] = coords
            app_state[sid]["activities"] = activity
            app_state[sid]["labels"] = label
            app_state[sid]["strands"] = strands
            #plausible_ots = []
            d = {
                "coords": coords, "activity": activity, "guide": guides, 
                "label": label, "strand": strands
            }
        return(
            html.Div(
                [
                    work_sector(d, effector, cart, sid), copyright()
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


def offtarget_sector(guides, effector, cart, sid):
    xaxis = "X"
    yaxis = "Y"
    sequences = [
        a+","+"activity:"+str(b)+","+"#offtargets:"+str(c)
            for a,b,c in zip(
                app_state[sid]["cart"], app_state[sid]["cart_activities"], 
                app_state[sid]["n_off"]
            )
    ]
    trace0 = go.Scatter(
        x=guides["X"],
        y=guides["Y"],
        mode="markers",
        marker=dict(
            size=32,
            showscale=False,
            color="blue"
        ),
        text=sequences,
        hoverinfo="text"
    )
    return(
        html.Div([
            html.Div(
                [
                    html.Div([dcc.Graph(
                        id='g3', 
                        figure={
                            'data': [trace0],
                            'layout': go.Layout(
                                xaxis={'title': "X"},
                                yaxis={'title': "Y"},
                                height=900
                            )
                        }
                    )], id="plot_space_2", className="nine columns"),
                    html.Hr(),
                    html.Center(html.H4("Guide cart")),
                    html.Div([guidecart(sid)], id="guide-cart-2", style={'fontSize': '12px'}),
                    html.Hr(),
                    html.Img(id="offtargetmm"),
                    html.Br(),
                    "Off-target mismatch composition",
                    html.Hr(),
                    html.Center([html.Button("Off-target report", id="off-repbutton")]),
                    html.Div([], id="off-report-link"),
                    html.Hr()
                ], className="row"
            )
        ])
    )

def work_sector(guides, effector, cart, sid):
    xaxis = "UMAP 1"
    yaxis = "UMAP 2"
    if guides:
        info = []
        for a,b,c,d in zip(guides["guide"], guides["activity"], guides["label"], guides["strand"]):
            info.append(
                str(a)+"; Activity:"+str(b)+"; Label:"+str(c)+"; Strand:"+str(d)+";")
        traces = [
            plotter.get_background(
                np.array(logic.background["coordinates"]), 
                np.array(logic.background["labels"]), 
                np.array(logic.background["labels"])
            ),
            plotter.get_guide(guides["coords"], guides["label"], guides["activity"], info)
        ]
        if len(app_state[sid]["cart"]) > 0:
            activity = app_state[sid]["activities"]
           # n_off = app_state[sid]["n_off"]
            labels = app_state[sid]["labels"]
            guides = app_state[sid]["guides"]
            strands = app_state[sid]["strands"]
            #act_n = app_state[sid]["act_n"]
            info = []
            in_cart = np.array([a.split(",")[0] for a in app_state[sid]["cart"]])
            for a,b,c,d in zip(guides, activity, labels, strands):
                gs = ",".join([a[0], str(a[1]), str(a[2]), a[3]])
                info.append(
                    gs+"; Activity:"+str(b)+"; Label:"+str(c)+"; Strand"+str(d)+";"# Off-targets:"+str(e)
                )
            gc = np.array([a.split(",")[0] in in_cart for a in info])
            coords = app_state[sid]["coords"]
            traces.append(
                plotter.get_guide_cart(
                    coords[gc], labels[gc], 
                    activity[gc], np.array(info)[gc]
                )
            )
    else:
        traces = [plotter.empty_plot()]
    return(
        html.Div([
            html.Div(
                [
                    html.Div([dcc.Graph(
                        id='g2', 
                        figure=plotter.make_plot(traces, xaxis=xaxis, yaxis=yaxis)
                    )], id="plot_space", className="nine columns"),
                    html.Div(
                        [
                            html.Center(html.H4("Guide cart")),
                            html.Div([guidecart(sid)], id="guide-cart", style={'fontSize': '12px'}),
                            html.Hr(),
                            html.Img(id='ontargetvicinity'),
                            html.Br(),
                            "On-target vicinity",
                            html.Hr(),
                            html.Center([html.Button("On-target report", id="on-repbutton")]),
                            html.Div([], id="on-report-link"),
                            html.Hr()
                        ], className="two columns")
                ], className="row"
            )
        ])
    )

def guidecart(sid):
    if sid in app_state:
        if len(app_state[sid]["cart"]) < 20:
            shown = [a for a in app_state[sid]["cart"]] 
        else: 
            shown = [a for a in app_state[sid]["cart"][0:19]]
            shown.append("+ "+str(len(app_state[sid]["cart"])-19)+" guides")
    else:
        shown = ["Cart is empty"]
    return(
        html.Ul([html.Li(a) for a in shown])
    )

@app.callback(
    Output("cart-store", "data"), [Input("g2", "clickData")], [State("session-id", "children")]
)
def change_guide_cart(clickData, sid):
    if clickData:
        cD = list(filter(lambda x: "text" in x, clickData["points"]))[0]
        #print(cD["text"])
        try:
            sequence = re.search("[ATGC]+,\d+,\d+,[ATGC]+", cD["text"])[0]
        except:
            sequence = re.search("[ATGC]+\',\s\d+,\s\d+,\s\'[ATGC]+", cD["text"])[0]
        finally:
            current = sequence.replace(" ", "").replace("'", "")
            if current in app_state[sid]["cart"]:
                app_state[sid]["cart"].pop(
                    app_state[sid]["cart"].index(current)
                )
            else:
                app_state[sid]["cart"].append(current)
            return({"cart": []})

@app.callback(
    Output("guide-cart", "children"),
    [Input("cart-store", "data")], [State("session-id", "children")]
)
def show_guide_cart(cart, sid):
    if cart:
        cart = cart["cart"]
    else:
        cart = []
    return(
        guidecart(sid)
    )
    
    
@app.callback(
    Output("ontargetvicinity", "src"),
    [Input("session-id", "children"), Input("cart-store", "data")]
)
def update_ontarget_logo(sid, cart):
    if cart:
        current_sequence = [a for a in app_state[sid]["cart"]][-1].split(",")[0]
        current_onehot = correct_order(onehot(current_sequence))
        current_representation = logic.predict(
            current_onehot.reshape(1, len(current_sequence)*4)
        )[0].reshape(1,64)
        current_coords = logic.transform(current_representation).reshape(2)
        current_d2b = np.apply_along_axis(
            lambda a: np.sum(np.abs(a-current_coords)), 1, logic.background["coordinates"]
        )
        top_k = list(sorted(np.arange(current_d2b.shape[0]), key=lambda x: current_d2b[x]))
        top_k = np.array(top_k[0:logic.k])
        top_k_seq = np.array(logic.background["sequences"])[top_k]
        img_fn = "".join([str(a) for a in np.random.choice(np.arange(10), 12)])+".png"
        logoworker.logo_of_list(top_k_seq, img_fn)
        return(static_image_route+img_fn)

@app.callback(
    Output("offtargetmm", "src"),
    [Input("session-id", "children"), Input("g3", "clickData")],
    #[State("mismatch-store", "data")]
)
def update_offtarget_logo(sid, clickData):#, n_mm):
    if clickData:
        cD = list(filter(lambda x: "text" in x, clickData["points"]))[0]
        current_sequence = [a for a in cD["text"].split(",")][0]
        current_pam = [a for a in cD["text"].split(",")][3]
        full_guide = current_sequence+current_pam[0] if not logic.pam_before else current_pam[0]+current_sequence
        img_fn = "".join([str(a) for a in np.random.choice(np.arange(10), 12)])+".png"
        logoworker.logo_of_list(app_state[sid]["offtargets"][full_guide], img_fn)
        return(static_image_route+img_fn)
    
@app.callback(
    Output("plot_space", "children"),
    [Input("cart-store", "data"), Input('session-id', 'children')],
    [State("plot_space", "children"), State("effector-store", "data")]
)
def plot_cart(cart, sid, old_plot, effector) :
    if effector:
        effector = effector["effector"]
    activity = app_state[sid]["activities"]
    #n_off = app_state[sid]["n_off"]
    labels = app_state[sid]["labels"]
    guides = app_state[sid]["guides"]
    strands = app_state[sid]["strands"]
    #act_n = app_state[sid]["act_n"]
    info = []
    in_cart = np.array([a.split(",")[0] for a in app_state[sid]["cart"]])
    for a,b,c,d in zip(guides, activity, labels, strands):
        gs = ",".join([a[0], str(a[1]), str(a[2]), a[3]])
        info.append(
            gs+"; Activity:"+str(b)+"; Label:"+str(c)+"; Strand"+str(d)#+"; Off-targets:"+str(e)
        )
    gc = np.array([a.split(",")[0] in in_cart for a in info])
    coords = app_state[sid]["coords"]
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
    Output("on-report-link", "children"),
    [Input("on-repbutton", "n_clicks")], 
    [State("session-id", "children"), State("cart-store", "data")]
)
def on_report(n, sid, cart):
    if n:
        s1 = [a[0] for a in app_state[sid]["guides"]]
        s2 = [a[1] for a in app_state[sid]["guides"]]
        s3 = [a[2] for a in app_state[sid]["guides"]]
        s4 = [a[3] for a in app_state[sid]["guides"]]
        data = pd.DataFrame(
            {
                "sequence": s1,
                "start": s2,
                "end": s3,
                "PAM": s4,
                "activity": app_state[sid]["activities"],
                "labels": app_state[sid]["labels"],
                "strands": ["+" if a == 1 else "-" for a in app_state[sid]["strands"]]
            }
        )
        data["chosen?"] = data["sequence"].isin([a.split(",")[0] for a in app_state[sid]["cart"]])
        pdf, fasta_all, fasta_chosen = get_on_html(data, logic.on_report)
        file = zip_file([pdf, fasta_all, fasta_chosen])
        return(html.Center(file_download_link(file)))
        
@app.callback(
    Output("off-report-link", "children"),
    [Input("off-repbutton", "n_clicks")], 
    [State("session-id", "children"), State("cart-store", "data")]
)
def off_report(n, sid, cart):
    if n:
        s1 = [a.split(",")[0] for a in app_state[sid]["cart"]]
        s2 = [a.split(",")[1] for a in app_state[sid]["cart"]]
        s3 = [a.split(",")[2] for a in app_state[sid]["cart"]]
        s4 = [a.split(",")[3] for a in app_state[sid]["cart"]]
        data = pd.DataFrame(
            {
                "sequence": s1,
                "start": s2,
                "end": s3,
                "PAM": s4,
                "activity": app_state[sid]["cart_activities"],
                "n_off": app_state[sid]["n_off"],
                "norm_n_off": app_state[sid]["norm_n_off"]
            }
        )
        pdf, fasta = get_off_html(data, logic.off_report)
        file = zip_file([pdf, fasta])
        return(html.Center(file_download_link(file)))
        
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
        
        
app.run_server(host='0.0.0.0', port=PORT, debug=False)
