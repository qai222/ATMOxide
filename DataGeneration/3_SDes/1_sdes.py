import os
from AnalysisModule.calculator.schema import HybridOxide
from AnalysisModule.calculator.descriptors import OxideStructureDC
from AnalysisModule.routines.util import save_jsonfile, read_jsonfile
from AnalysisModule.routines.data_settings import SDPATH
from tqdm import tqdm
import logging
from joblib import Parallel, delayed
from AnalysisModule.routines.util import tqdm_joblib, get_mo_compositions_from_csd, get_mo_compositions_from_sslist

import pandas as pd

thisdir = os.path.dirname(os.path.abspath(__file__))
identifiers = pd.read_csv("{}/../1_ChemicalDiagramSearch/4_bucurate.csv".format(thisdir))["identifier"].tolist()

split_data_folder = SDPATH.split_data
structural_descriptor_fold = SDPATH.sdes_data
split_data_jsons = SDPATH.split_jsons


def compcheck(csdformula, clean_sslist):
    moc = get_mo_compositions_from_csd(csdformula)
    moc_clean = get_mo_compositions_from_sslist(clean_sslist)
    if moc_clean.num_atoms == 0 or moc.num_atoms == 0:
        # raise DescriptorError("composition check failed: {} vs {}".format(moc_clean, moc))
        logging.warning("composition check failed: {} vs {}".format(moc_clean, moc))
        return False
    elif moc_clean.reduced_composition != moc.reduced_composition:
        return False
    return True


def cal_structural_descriptors(identifier):
    split_json_file = "{}/{}-split.json".format(split_data_folder, identifier)
    des_json_file = "{}/{}-sdes.json".format(structural_descriptor_fold, identifier)
    # ho_jsone_file = "{}/{}-hybridoxide.json".format(structural_descriptor_fold, identifier)
    if os.path.isfile(des_json_file):
        if os.path.getsize(des_json_file) > 0:
            print('found json, skip: {}'.format(identifier))
            return

    if not os.path.isfile(split_json_file) or os.path.getsize(split_json_file) == 0:
        print('split json not found: {}'.format(identifier))
        return

    logfile = '{}/{}-sdes.log'.format(structural_descriptor_fold, identifier)

    results = read_jsonfile(split_json_file)
    formula = results["csd_formula"]
    clean_ss_list = results['clean_structure_substructure_list']
    disorder_ss_list = results['disorder_structure_substructure_list']

    for handler in logging.root.handlers:
        logging.root.removeHandler(handler)
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO, filemode='w', filename=logfile)
    logging.captureWarnings(True)
    logging.info("-- working on: {}".format(identifier))

    if os.path.isfile(des_json_file):
        des_data = read_jsonfile(des_json_file)
    else:
        des_data = None
    try:
        if compcheck(formula, clean_ss_list):
            logging.info("compcheck good, using clean ss list")
            ho = HybridOxide.from_substructures(clean_ss_list, identifier, None, strip_alkali=False)
        else:
            logging.info("compcheck failed, using disorder ss list")
            ho = HybridOxide.from_substructures(disorder_ss_list, identifier, None, strip_alkali=False)
            # ho.oxide_structure.to("cif", "{}/{}-oxide.cif".format(structural_descriptor_fold, identifier))
            # ho.total_structure.to("cif", "{}/{}-total.cif".format(structural_descriptor_fold, identifier))
        ods = OxideStructureDC(ho.oxide_structure)
        ds = ods.get_descriptors(updatedict=des_data, force_recal=False)
        save_jsonfile(ds, des_json_file)
        # from pprint import pprint
        # pprint(ds)
        # save_jsonfile(ho, ho_jsone_file)
        # with open('{}/{}-des.yml'.format(structural_descriptor_fold, csdmeta.identifier), 'w') as outfile:
        #     yaml.dump(ho.as_dict(), outfile, default_flow_style=False)
        logging.info("-- SUCCESS!")
    except Exception as e:
        logging.exception("-- FAILED!: " + str(e))


with tqdm_joblib(tqdm(desc="structural des", total=len(identifiers))) as progress_bar:
    Parallel(n_jobs=6)(delayed(cal_structural_descriptors)(i) for i in identifiers)
