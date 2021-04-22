import ast
from copy import deepcopy

import matplotlib.pyplot as plt
import pandas as pd
from bokeh.io import output_file, show
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure
from bokeh.sampledata.periodic_table import elements
from bokeh.transform import dodge
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, to_hex
from AnalysisModule.routines.util import MDefined, NMDefined, AllElements

"""
plot a periodic table with a dataframe of element occurrence in crystal structures
based on 
https://github.com/arosen93/ptable_trends
https://docs.bokeh.org/en/latest/docs/gallery/periodic.html
"""

def get_accu_abc(mode):
    dfpredall = pd.read_csv("../../DimPredict/7_PbyElement/all_abc.csv")
    pred_records = dfpredall.to_dict("records")
    task_cols = [c for c in dfpredall.columns if c.startswith("task_") and c.endswith(mode)]
    counter = dict()
    for e in AllElements:
        total_prediction = 0
        right_prediction = 0
        for r in pred_records:
            record_elements = ast.literal_eval(r["elements"])
            ms = [ele_symbol for ele_symbol in record_elements if ele_symbol in MDefined]
            nms = [ele_symbol for ele_symbol in record_elements if ele_symbol in NMDefined]
            if e not in nms+ms:
                continue
            for task_col in task_cols:
                total_prediction += 1
                if r[task_col]:
                    right_prediction +=1
        if total_prediction == 0:
            continue
        counter[e] = round(right_prediction/total_prediction * 100, 1)
    zero_elements = ["H", ]
    for e in zero_elements:
        counter.pop(e, None)
    return counter

def get_accu():
    dfpredall = pd.read_csv("../../DimPredict/7_PbyElement/all.csv")
    pred_records = dfpredall.to_dict("records")
    pred_cols = [c for c in dfpredall.columns if c.startswith("dim_pred_")]
    counter = dict()
    for e in AllElements:
        total_prediction = 0
        right_prediction = 0
        for r in pred_records:
            record_elements = ast.literal_eval(r["elements"])
            ms = [ele_symbol for ele_symbol in record_elements if ele_symbol in MDefined]
            nms = [ele_symbol for ele_symbol in record_elements if ele_symbol in NMDefined]
            if e not in nms+ms:
                continue
            right_dim = r["dimension"]
            for pred_col in pred_cols:
                total_prediction += 1
                if r[pred_col] == right_dim:
                    right_prediction +=1
        if total_prediction == 0:
            continue
        counter[e] = round(right_prediction/total_prediction * 100, 1)
    zero_elements = ["H", ]
    for e in zero_elements:
        counter.pop(e, None)
    return counter


def get_occurance():
    dfin = pd.read_csv("../../DataGeneration/5_SimpleInput/input.csv")
    counter = dict()
    for e in AllElements:
        count = 0
        for es in dfin["elements"]:
            if e in ast.literal_eval(es):
                count += 1
        counter[e] = count
    zero_elements = ["H", ]
    for e in zero_elements:
        counter.pop(e, None)
    return counter


def get_elements_data(elements, occu_counter, accu_dict):
    elements = deepcopy(elements)  # a dataframe
    saoto_count = []
    saoto_accu = []
    for e in elements["symbol"]:
        try:
            saoto_count.append(int(occu_counter[e]))
        except KeyError:
            saoto_count.append(0)
        try:
            saoto_accu.append(accu_dict[e])
        except KeyError:
            saoto_accu.append(0)
        # try:
        #     saoto_count.append(str(occu_counter[e]))
        # except KeyError:
        #     saoto_count.append("")
        # try:
        #     saoto_accu.append(str(accu_dict[e]))
        # except KeyError:
        #     saoto_accu.append("")
    elements["saoto_count"] = saoto_count
    elements["saoto_accu"] = saoto_accu

    # La, Ac, H
    count = 0
    for i in range(56, 70):
        elements.loc[i, "period"] = "La"
        elements.loc[i, "group"] = str(count + 4)
        count += 1

    count = 0
    for i in range(88, 102):
        elements.loc[i, "period"] = "Ac"
        elements.loc[i, "group"] = str(count + 4)
        count += 1

    elements.loc[0, "group"] = 17

    return elements

import matplotlib.colors as colors
import  numpy as np
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def get_color_scale(elements, field, fieldmax, fieldmin):
    data = elements[field]
    pltcmap = "Wistia"
    norm = Normalize(vmin=fieldmin, vmax=fieldmax)
    cmap=plt.get_cmap(pltcmap)
    cmap = truncate_colormap(cmap, 0.2, 1.0)
    color_scale = ScalarMappable(norm=norm, cmap=cmap).to_rgba(data, alpha=None)
    return color_scale


def ptplot(elements: pd.DataFrame, outfile="elements.html", colorwhat="accu"):
    output_file(outfile)
    # Define number of and groups
    period_label = ['1', '2', '3', '4', '5', '6', '7']
    group_range = [str(x) for x in range(1, 19)]

    period_label.append('blank')
    period_label.append('La')
    period_label.append('Ac')

    # Define color for blank entries
    blank_color = '#c4c4c4'
    color_list = []
    for i in range(len(elements)):
        color_list.append(blank_color)

    # Compare elements in dataset with elements in periodic table
    if colorwhat == "occu":
        colorscale = get_color_scale(df_elements, "saoto_count", 800, 0)
        for i, symbol in enumerate(elements.symbol):
            element_entry = elements.iloc[i]
            if element_entry["saoto_count"] == 0:
                continue
            if symbol in ["P", "O", ]:
                color_list[i] = '#ff0000'
            else:
                color_list[i] = to_hex(colorscale[i])
    else:
        colorscale = get_color_scale(df_elements, "saoto_accu", 100, 0)
        for i, symbol in enumerate(elements.symbol):
            element_entry = elements.iloc[i]
            if element_entry["saoto_accu"] == 0:
                continue
            else:
                color_list[i] = to_hex(colorscale[i])

    # replace zeros with empty string
    accu_repr = []
    for accu in elements["saoto_accu"]:
        if accu == 0:
            accu_repr.append("")
        else:
            faccu = "{:.1f}".format(accu)
            if faccu == "100.0":
                faccu = "100"
            accu_repr.append(faccu)
    occu_repr = []
    for occu in elements["saoto_count"]:
        if occu == 0:
            occu_repr.append("")
        else:
            occu_repr.append(occu)
    elements["saoto_accu"] = accu_repr
    elements["saoto_count"] = occu_repr

    # source for plot
    source = ColumnDataSource(
        data=dict(
            group=[str(x) for x in elements['group']],
            period=[str(y) for y in elements['period']],
            sym=elements['symbol'],
            saoto_count=elements['saoto_count'],
            saoto_accu=elements['saoto_accu'],
            atomic_number=elements['atomic number'],
            type_color=color_list
        )
    )
    p = figure(x_range=group_range, y_range=list(reversed(period_label)), plot_width=480, plot_height=250,
               toolbar_location=None, )
    p.outline_line_color = None
    p.toolbar_location = 'above'
    p.rect('group', 'period', 0.9, 0.9, source=source,
           color='type_color')
    p.axis.visible = False
    p.output_backend = "svg"
    text_props = {
        'source': source,
        'angle': 0,
        'text_align': 'right',
        'text_baseline': 'middle'
    }
    symbol_props = {
        'source': source,
        'angle': 0,
        'text_align': 'left',
        'text_baseline': 'middle'
    }
    x_symbol = dodge("group", -0.4, range=p.x_range)
    y_symbol = dodge("period", 0.0, range=p.y_range)
    p.text(x=x_symbol, y=y_symbol, text='sym',
           text_font_style='bold',
           text_font_size='4.5pt',
           text_color="white",
           **symbol_props)

    x_count = dodge("group", 0.41, range=p.x_range)
    y_count = dodge("period", -0.2, range=p.y_range)
    p.text(x=x_count, y=y_count, text='saoto_count',
           text_font_size='4.8pt',
           text_color="purple",
           **text_props)

    x_accu = dodge("group", 0.41, range=p.x_range)
    y_accu = dodge("period", 0.15, range=p.y_range)
    p.text(x=x_accu, y=y_accu, text='saoto_accu',
           text_font_size='4.8pt',
           text_color="blue",
           **text_props)

    p.grid.grid_line_color = None
    xys = [(12, 9), (12, 8), (13, 8), (14, 8), (14, 7), (14, 6), (16, 6), (16, 5), (17, 5), (17, 10)]
    p.line([xy[0] for xy in xys], [xy[1] for xy in xys], line_width=4, line_color="red")
    show(p)

if __name__ == '__main__':
    occu_counter = get_occurance()
    for mode in ["a", "b", "c"]:
        accu_counter = get_accu_abc(mode)
        print(accu_counter)
        df_elements = get_elements_data(elements, occu_counter, accu_counter)
        ptplot(df_elements, outfile="elements_{}.html".format(mode))

    # accu_counter = get_accu()
    # print(accu_counter)
    # df_elements = get_elements_data(elements, occu_counter, accu_counter)
    # ptplot(df_elements, outfile="elements.html")
