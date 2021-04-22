import os
from AnalysisModule.prepare.split import Splitter, Structure
from AnalysisModule.routines.util import save_jsonfile, read_jsonfile
from tqdm import tqdm
import logging
from AnalysisModule.prepare.saentry import SaotoEntry
from joblib import Parallel, delayed
from AnalysisModule.routines.util import tqdm_joblib
from AnalysisModule.routines.data_settings import SDPATH

thisdir = os.path.dirname(os.path.abspath(__file__))
clean_data_folder = SDPATH.clean_data
split_data_folder = SDPATH.split_data
# clean_structure_jsons = sorted(list(glob.glob("{}/*.json".format(clean_data_folder))))
clean_structure_jsons = SDPATH.clean_jsons

os.chdir(split_data_folder)


def split_clean_structure(saentry_json):
    saentry: SaotoEntry = read_jsonfile(saentry_json)
    formula = saentry.details["formula"]
    identifier = saentry.identifier
    logfile = '{}-split.log'.format(identifier)
    results_jsonfile = '{}-split.json'.format(identifier)

    if os.path.isfile(results_jsonfile):
        if os.path.getsize(results_jsonfile) > 0:
            print('found json, skip: {}'.format(identifier))
            return

    for handler in logging.root.handlers:
        logging.root.removeHandler(handler)
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO, filemode='w', filename=logfile)
    logging.captureWarnings(True)


    clean_structure = Structure.from_dict(saentry.clean_structure)
    disordered_structure = Structure.from_dict(saentry.disodered_structure)

    possible_deuterium = set()
    for sp in disordered_structure.species:
        symbol = sp.symbol
        if symbol.startswith("D") and symbol not in ("Ds", "Db", "Dy"):
            possible_deuterium.add(symbol)
    for s in possible_deuterium:
        clean_structure.replace_species({s: "H"})
        disordered_structure.replace_species({s: "H"})
    try:
        clean_s, clean_ss_list = Splitter.split_structure_ocelot(clean_structure, cifname=None)  # unwrap is done here
        disorder_s, disorder_ss_list = Splitter.split_structure_ocelot(disordered_structure, cifname=None)
        results = {
            'identifier': identifier,
            'csd_formula': formula,
            'clean_structure': clean_s,
            'clean_structure_substructure_list': clean_ss_list,
            'disorder_structure': disorder_s,
            'disorder_structure_substructure_list': disorder_ss_list,
        }
        save_jsonfile(results, results_jsonfile)
    except Exception as e:
        logging.info("split failed!")
        logging.exception(str(e))

if __name__ == '__main__':

    # split_clean_structure(clean_structure_jsons[0])
    with tqdm_joblib(tqdm(desc="split", total=len(clean_structure_jsons))) as progress_bar:
        Parallel(n_jobs=4)(delayed(split_clean_structure)(i) for i in clean_structure_jsons)
    # 4h30min
