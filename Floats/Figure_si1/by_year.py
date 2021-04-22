import pandas as pd
from AnalysisModule.routines.util import read_jsonfile
df = pd.read_csv("../../DataGeneration/5_SimpleInput/input.csv")
i2y = read_jsonfile("identifier_labeled_year.json")
df['year'] = df.apply(lambda x: i2y[x["identifier"]], axis=1)


def x_used_by_year(x:str, year:int):
    subdf = df[df["year"]<= year][x]
    # print(subdf)
    return sorted(set(subdf))

def x_new_in_year(x:str, year:int):
    x_used = x_used_by_year(x, year-1)
    x_this_year = set(df[df["year"]== year][x])
    return x_this_year.difference(set(x_used))

xs = sorted(set(df["year"]))[:-1]

def get_cumulative_y(y:str):
    return [len(x_used_by_year(y, year)) for year in xs]

def get_delta_y(y:str):
    return [len(x_new_in_year(y, year)) for year in xs]


import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})

title_dict = {
    "identifier": "structures",
    "smiles": "TA",
    "elements": "EP",
}

def plot_by_field(field):
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_title('Number of {}'.format(title_dict[field]))
    ax1.set_xlabel('Year', color="k")  # we already handled the x-label with ax1
    ax1.set_ylabel('Cumulative Count', color=color)  # we already handled the x-label with ax1
    ax1.plot(xs, get_cumulative_y(field), color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    color = 'tab:blue'
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_xlabel('Year', color=color)  # we already handled the x-label with ax1
    ax2.set_ylabel('Count per year', color=color)
    ax2.plot(xs, get_delta_y(field), color=color, ls=":")
    ax2.tick_params(axis='y', labelcolor=color)

    from pprint import pprint
    print(field, ":")
    pprint(list(zip(xs, get_delta_y(field), get_cumulative_y(field))))

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    fig.savefig("by_year_{}.png".format(field), dpi=300)
    fig.savefig("by_year_{}.eps".format(field))
    plt.clf()

field = "elements"
plot_by_field(field)
field = "smiles"
plot_by_field(field)
field = "identifier"
plot_by_field(field)
