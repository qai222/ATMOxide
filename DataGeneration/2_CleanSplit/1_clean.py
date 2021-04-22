import os
from AnalysisModule.prepare.csd_exporter import Entry, get_CsdEntry, CsdEntry2SaotoEntry
from AnalysisModule.routines.util import save_jsonfile
from tqdm import tqdm
import logging
from AnalysisModule.routines.util import tqdm_joblib
from AnalysisModule.routines.data_settings import SDPATH
from joblib import Parallel, delayed

thisdir = os.path.dirname(os.path.abspath(__file__))
os.chdir(SDPATH.clean_data)


def clean_identifier(identifier: str):
    logfile = '{}-clean.log'.format(identifier)
    results_jsonfile = '{}-clean.json'.format(identifier)

    if os.path.isfile(results_jsonfile):
        if os.path.getsize(results_jsonfile) > 0:
            print('found json, skip: {}'.format(identifier))
            return

    for handler in logging.root.handlers:
        logging.root.removeHandler(handler)
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO, filemode='w', filename=logfile)
    logging.captureWarnings(True)

    e = get_CsdEntry(identifier)
    # raw_string = e.crystal.to_string('cif')
    # with open('{}-raw.cif'.format(identifier), 'w') as f:
    #     f.write(raw_string)
    try:
        sae = CsdEntry2SaotoEntry(e, None,
                                  source_object_name="disordered_molecule",
                                  trust_csd_has_disorder=True,
                                  disorder_clean_strategy="remove_labels_with_nonword_suffix",
                                  )
    except Exception as e:
        logging.exception(str(e))
        return

    save_jsonfile(sae, results_jsonfile)
    # sae.clean_structure.to('cif', '{}-clean.cif'.format(identifier))


if __name__ == '__main__':
    import pandas as pd
    identifiers = pd.read_csv("{}/../1_ChemicalDiagramSearch/4_bucurate.csv".format(thisdir))["identifier"].tolist()
    e: Entry
    with tqdm_joblib(tqdm(desc="cleansplit", total=len(identifiers))) as progress_bar:
        Parallel(n_jobs=4)(delayed(clean_identifier)(i) for i in identifiers)
