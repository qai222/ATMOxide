import base64
import os

import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import plotly.graph_objects as go
import pickle
import typing
import pathlib

def load_pkl(fn: typing.Union[str, pathlib.Path]):
    with open(fn, "rb") as f:
        o = pickle.load(f)
    return o

dims = (0, 1, 2, 3)
color = ["blue", "red", "green", "brown"]
color_code = dict(zip(dims, color))

import dash
from dash.dependencies import Input, Output

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
# logit_result = yaml_fileread("../logistic.yml")
logit_result: dict
logit_result = load_pkl("../clf2d/logistic.pkl")
chemical_system_list = list(k for k in logit_result.keys() if logit_result[k] is not None)

def get_max_s(chemical_system="C-Co-O"):
    epginfo = logit_result[chemical_system]
    max_s_reached = epginfo["max_s_reached"]
    max_s = epginfo["max_s"]
    baseline = epginfo["baseline"]
    return max_s, max_s_reached, baseline


def get_feature_combinations(chemical_system="C-Co-O"):
    epginfo = logit_result[chemical_system]
    results = epginfo["comb_results"]
    combs = []
    for r in results:
        used_features = "{} -- {}".format(*r["used_features"])
        combs.append(used_features)
    return combs


def gen_gofig(chemical_system="C-Co-O", used_features=None):
    epginfo = logit_result[chemical_system]
    max_s = epginfo["max_s"]
    xx = epginfo["xx"]
    yy = epginfo["yy"]
    baseline = epginfo["baseline"]
    refcodes = epginfo["refcodes"]
    dims = epginfo["dims"]
    amines = epginfo["amines"]
    results = epginfo["comb_results"]
    plot_data = {}
    for r in results:
        used_features_this = "{} -- {}".format(*r["used_features"])
        if used_features_this != used_features:
            continue
        score = r["score"]
        xs = r["xs"]
        ys = r["ys"]
        Z = r["Z"]
        plot_data["score"] = score
        plot_data["xs"] = xs
        plot_data["ys"] = ys
        plot_data["zs"] = dims
        plot_data["zz"] = Z
        plot_data["xx"] = xx
        plot_data["yy"] = yy
        break

    xname, yname = used_features.split(" -- ")
    score = plot_data["score"]

    def get_heatmap(plot_data):
        xx = plot_data["xx"]
        yy = plot_data["yy"]
        zz = plot_data["zz"]
        xx = np.array(xx).flatten()
        yy = np.array(yy).flatten()
        zz = np.array(zz).flatten()
        zz = [color_code[z] for z in zz]
        marker = {"color": zz, "symbol": "square", "size": 8.5}
        heatmap = go.Scatter(x=xx, y=yy, marker=marker, mode="markers", hoverinfo="skip", opacity=0.5)
        return heatmap

    def get_scatter(plot_data):
        xs = plot_data["xs"]
        ys = plot_data["ys"]
        zs = plot_data["zs"]
        zs = [color_code[z] for z in zs]
        marker = {"color": zs, "symbol": "circle", "size": 15}
        texts = []
        for i, refcode in enumerate(refcodes):
            text = "<b>refcode: {}</b><br>".format(refcode)
            text += "<b>amine: {}</b><br>".format(amines[i])
            text += "<b>dim: {}</b>".format(dims[i])
            texts.append(text)
        scatter = go.Scatter(
            x=xs, y=ys, marker=marker, mode="markers", opacity=1.0,
            hovertemplate=
            # '<b>x</b>: %{x:.2f}<br>'+
            # '<b>y</b>: %{y:.2f}<br>'+
            '%{text}<extra></extra>',
            text=texts,
            customdata=[{"amine": amines[i], "refcode": refcodes[i]} for i in range(len(amines))],
        )
        return scatter

    heatmap = get_heatmap(plot_data)
    scatter = get_scatter(plot_data)

    layout = {
        "width": 600,
        "height": 600,
        "title": "# of structures: {}, # of amines: {}<br>max/logit/base: {:.2f}/{:.2f}/{:.2f}".format(len(refcodes),
                                                                                                       len(set(amines)),
                                                                                                       max_s, score,
                                                                                                       baseline),
        "yaxis_range": [-0.03, 1.03],
        "xaxis_range": [-0.03, 1.03],
        "xaxis_fixedrange": True,
        "yaxis_fixedrange": True,
        'xaxis_showgrid': False,
        'yaxis_showgrid': False,
        'xaxis_title': xname,
        'yaxis_title': yname,
        "dragmode": False,
        "showlegend": False,
    }
    fig = go.Figure(data=[heatmap, scatter])
    fig.update_layout(**layout)
    fig.update_layout(clickmode='event+select')

    return fig


app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    html.P("chemical system:"),
    dcc.RadioItems(
        id="chemical_system-radio",
        options=[
            {"label": "{} {:.2f}/{:.2f}/{:.2f} (max score/max score reached/baseline)".format(k, *get_max_s(k)),
             "value": k} for k in chemical_system_list
        ],
        value=chemical_system_list[0]
    ),
    html.P("feature combination:"),
    dcc.Dropdown(
        id='feature_combination-dropdown',
    ),

    html.Div(
        [
            html.Div(
                children=[
                    html.H3("2-feature logistic regression"),
                    dcc.Graph(
                        id="maingraph",
                        # className = "two-thirds column",
                    ),
                ],
                # style={"width": "40%", "float": "left", },
                style={"width": "40%", "float": "left", "min-width":"600px"},
                # className="six columns"
                # style={"display": "inline-block"}
            ),
            html.Div(
                id="amine_vis-div",
                children=[],
                style={"width": "60%", "margin-top":"2em"},
                # style={"width": "60%", "margin-left": "40%"},
                # style={"display": "inline-block"},
                # className="six columns"
            ),

        ],
        style={"width": "100%", "display":"flex"}
        # className="row",
    ),

    # html.Img(
    #     id="amine_vis-img",
    #     width=200,
    #     height=200,
    #     style={"display": "inline-block"}
    # )

]
)


@app.callback(
    Output("feature_combination-dropdown", "options"),
    [Input("chemical_system-radio", "value")])
def available_feature_combinations(chemical_system):
    combs = get_feature_combinations(chemical_system)
    return [{"label": fc, "value": fc} for fc in combs]


@app.callback(
    Output("maingraph", "figure"),
    [Input("feature_combination-dropdown", "value"), Input("chemical_system-radio", "value")])
def push_new_maingraph(used_features, chemical_system):
    if used_features is None:
        return {}
    mainfig = gen_gofig(chemical_system, used_features)
    return mainfig


# @app.callback(
#     Output('amine_vis-img', 'src'),
#     Input('maingraph', 'clickData'))
# def display_click_amine(clickData):
#     if not isinstance(clickData, dict):
#         svgfilepath = os.path.abspath("amines/{}.svg".format("dummy_mol"))
#     else:
#         point = clickData["points"][-1]
#         data = point["customdata"]
#         # amine = data["amine"]
#         refcode = data["refcode"]
#         svgfilepath = os.path.abspath("amines/{}.svg".format(refcode))
#     encoded = base64.b64encode(open(svgfilepath, 'rb').read())
#     svg = 'data:image/svg+xml;base64,{}'.format(encoded.decode())
#     return svg

@app.callback(
    Output('amine_vis-div', 'children'),
    Input('maingraph', 'selectedData'))
def display_selected_data(selectedData):
    if not isinstance(selectedData, dict) or len(selectedData) == 0:
        svgfilepaths = [os.path.abspath("{}.svg".format("dummy_mol"))]
    else:
        svgfilepaths = []
        points = selectedData["points"]
        for point in points:
            data = point["customdata"]
            # amine = data["amine"]
            refcode = data["refcode"]
            svgfilepath = os.path.abspath("amines/{}.svg".format(refcode))
            svgfilepaths.append(svgfilepath)
    children = []
    for svgfilepath in svgfilepaths:
        ref = os.path.basename(svgfilepath)[:-4]
        url = "https://www.ccdc.cam.ac.uk/structures/Search?Ccdcid={}".format(ref)
        encoded = base64.b64encode(open(svgfilepath, 'rb').read())
        svg = 'data:image/svg+xml;base64,{}'.format(encoded.decode())
        img = html.Img(
            src=svg, width=200, height=200,
            # style={"display": "inline-block"}
        )
        child = html.Div(
            [
                img,
                dcc.Link(ref, href=url, style={"font-size": "2em"})
            ],
            style={"width": 200, "display": "inline-block", "border-width": "2px", "border-style": "dotted",
                   "margin": "3px"}
        )
        children.append(child)
    return children


if __name__ == '__main__':
    # app.run_server(debug=True)
    app.run_server("0.0.0.0", debug=False, threaded=True)
