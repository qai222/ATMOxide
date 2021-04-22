import pandas as pd
from tqdm import tqdm
from AnalysisModule.routines.util import save_jsonfile
from ccdc.io import EntryReader, Entry

# dict identifier -> publication year
df = pd.read_csv("../5_SimpleInput/input.csv")
CSD_READER = EntryReader('CSD')
identifer2year = dict()
for i in tqdm(df["identifier"]):
    entry = CSD_READER.entry(i)
    entry:Entry
    identifer2year[i] = entry.publication.year
save_jsonfile(identifer2year, "identifier_labeled_year.json")
