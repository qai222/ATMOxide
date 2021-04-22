import hashlib
import itertools
import json
import logging
import pathlib
import pickle
import random
import re
import shutil
import typing
from copy import deepcopy
from operator import eq

import networkx as nx
import networkx.algorithms.isomorphism as iso
import pandas as pd
import yaml
from monty.json import MontyDecoder, MontyEncoder
from networkx.readwrite import json_graph
from pymatgen.core.composition import Composition
from pymatgen.core.periodic_table import _pt_data
from pymatgen.core.structure import Structure
from rdkit import Chem

from AnalysisModule.routines.pbc import AtomicRadius_string

AllElements = set(_pt_data.keys())


def load_pkl(fn: typing.Union[str, pathlib.Path]):
    with open(fn, "rb") as f:
        o = pickle.load(f)
    return o


def save_pkl(o, fn: typing.Union[str, pathlib.Path]):
    with open(fn, "wb") as f:
        pickle.dump(o, f)


def hash_structure(structure, hashmethod=hashlib.sha256):
    # copied from ocelot.schema.configuration.hashconfig
    # maybe worth to implement molecular comps hash
    latt = structure.lattice
    latt_str = '_'.join(['{:.3f}'] * 9)
    latt_str = latt_str.format(*latt.matrix.flatten())
    comp_str = structure.composition.__repr__()
    combine = '---'.join([latt_str, comp_str])
    return hashmethod(combine.encode('utf-8')).hexdigest()


def createRandomSortedList(num, start, end):
    arr = []
    tmp = random.randint(start, end)
    for x in range(num):

        while tmp in arr:
            tmp = random.randint(start, end)
        arr.append(tmp)
    arr.sort()
    return arr


NORQUIST_identifiers = [
    'HUWDIN',
    'HUWDOT',
    'HUWDUZ',
    'ORIPUB',
    'ORIQAI',
    'QOXCEN',
    'QOXCIR',
    'QOXCOX',
    'QOXCUD',
    'QOXDAK',
    'RAHBEI',
    'REKKOJ',
    'REKKUP',
    'REKLAW',
    'REKLEA',
    'REKLIE',
]

NORQUIST_identifiers_all = [
    'ABAMIC',
    'ASEQIA',
    'ASEQOG',
    'ASEQUM',
    'ASERAT',
    'ASEREX',
    'ASERIB',
    'AWEFAK01',
    'AWEFAK',
    'AWEFIS01',
    'AWEFIS',
    'AXOHOL',
    'AXOHUR',
    'CEHQAJ',
    'CEHQEN',
    'CEHQIR',
    'DINCIP',
    'ENIFIR',
    'ENIFOX',
    'ENIFUD',
    'ENIGAK',
    'HIKTIG',
    'HIKTOM',
    'HIKTUS',
    'HIKVAA',
    'HIKVEE',
    'HIKVII',
    'HIKVOO',
    'HIKVUU',
    'HIKWAB',
    'HUWDIN',
    'HUWDOT',
    'HUWDUZ',
    'IKELOA',
    'JOFMID',
    'JOFMOJ',
    'ORIPUB',
    'ORIQAI',
    'QIFNOK',
    'QIFNUQ',
    'QIFPAY',
    'QOXCEN',
    'QOXCIR',
    'QOXCOX',
    'QOXCUD',
    'QOXDAK',
    'REKKOJ',
    'REKKUP',
    'REKLAW',
    'REKLEA',
    'REKLIE',
    'YOTHOG',
    'YOTHUM',
]


def yaml_filedump(o, fn: typing.Union[str, pathlib.Path]):
    with open(fn, 'w') as outfile:
        yaml.dump(o, outfile, default_flow_style=False)


def yaml_fileread(fn: typing.Union[str, pathlib.Path]):
    with open(fn, 'r') as stream:
        o = yaml.safe_load(stream)
    return o


def read_jsonfile(p: typing.Union[str, pathlib.Path], decoder=MontyDecoder):
    with open(p, "r") as f:
        r = json.load(f, cls=decoder)
    return r


def save_jsonfile(o, p: typing.Union[str, pathlib.Path], encoder=MontyEncoder):
    with open(p, "w") as f:
        json.dump(o, f, cls=encoder)


def movefile(what, where):
    """
    shutil operation to move
    :param what:
    :param where:
    :return:
    """
    try:
        shutil.move(what, where)
    except IOError:
        os.chmod(where, 777)
        shutil.move(what, where)


def removefile(what):
    try:
        os.remove(what)
    except OSError:
        pass


def copyfile(what, where):
    """
    shutil operation to copy
    :param what:
    :param where:
    :return:
    """
    try:
        shutil.copy(what, where)
    except IOError:
        os.chmod(where, 777)
        shutil.copy(what, where)


def createdir(directory):
    """
    mkdir
    :param directory:
    :return:
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def csdformula2pmg_composition(formula: str, neutralize_hcations=False):
    # do we need to worry about brackets?
    comps = formula.split(",")
    pmg_comps = []
    for comp in comps:
        comp_formula = " ".join(re.findall(r"[A-Z]{1}[a-z]{0,1}\d*", comp))
        if len(comp_formula.strip()) == 0:
            continue
        try:
            pmg_composition = Composition(comp_formula)
        except ValueError:
            pmg_composition = Composition(comp_formula.replace("D", "H"))
        if not (PmgComp_Ematch(pmg_composition, "C N H") or PmgComp_Ematch(pmg_composition, "C H")):
            # forbid neutralize if it's not an amine-like
            do_neutralize = False
        elif neutralize_hcations:
            do_neutralize = True
        else:
            do_neutralize = False

        if do_neutralize:
            try:
                charge_string = re.findall(r"[0-9]{0,3}(?:\+|\-){1}", comp)[0]
            except IndexError:
                charge_string = ""
            if "+" in charge_string and "h" in comp_formula.lower():
                try:
                    comp_charge = int(re.findall(r"\d*", charge_string)[0])
                except IndexError:
                    comp_charge = 1
                pmg_composition = pmg_composition - Composition("H") * comp_charge
            elif "-" in charge_string and "h" in comp_formula.lower():
                try:
                    comp_charge = int(re.findall(r"\d*", charge_string)[0])
                except IndexError:
                    comp_charge = 1
                pmg_composition = pmg_composition + Composition("H") * comp_charge

        pmg_comps.append(pmg_composition)
    return pmg_comps


def get_no_hydrogen_composition(c: Composition):
    nc = c - c["H"] * Composition("H")
    return nc


def PmgComp_Ematch(c: Composition, elements: str = "C N"):
    """
    element check for composition, comp_elements(c, "C N H") is True == only has "C N H"
    """
    return set(c.elements) == set(Composition(elements).elements)


def strip_elements_from_formula(f: str, etype="alkali"):
    if etype == "alkali":
        es = ("Na", "Li", "K", "Rb", "Ce")
    else:
        es = [etype]
    for e in es:
        f = re.sub(r"{}\d*".format(e), "", f)
    return f


def strip_elements(s: Structure, etype="alkali"):
    new_sites = []
    sites_to_be_removed = []
    if isinstance(etype, str):
        for site in s:
            if site.species.contains_element_type(etype):
                sites_to_be_removed.append(site)
                continue
            ns = deepcopy(site)
            new_sites.append(ns)
    elif isinstance(etype, list):
        for site in s:
            if site.species_string in etype:
                continue
            ns = deepcopy(site)
            new_sites.append(ns)

    try:
        remove_labels = [site.properties["label"] for site in sites_to_be_removed]
        for ns in new_sites:
            new_nninfos = []
            nninfos = ns.properties["nn"]
            for nninfo in nninfos:
                if nninfo[0] not in remove_labels:
                    new_nninfos.append(nninfo)
            ns.properties["nn"] = new_nninfos
        return Structure.from_sites(new_sites)
    except Exception as e:
        logging.exception(str(e))
        return Structure.from_sites(new_sites)


import contextlib, joblib


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def rdkitmol2comp(m: Chem.Mol, addh=True):
    if addh:
        mol = Chem.AddHs(m)
    else:
        mol = m
    c = Composition("")
    a: Chem.Atom
    for a in mol.GetAtoms():
        c += Composition(a.GetSymbol())
    return c


def rdmol2nxg(rdmol: Chem.Mol):
    g = nx.Graph()
    for atom in rdmol.GetAtoms():
        g.add_node(atom.GetIdx(),
                   symbol=atom.GetSymbol(),
                   )
    for bond in rdmol.GetBonds():
        g.add_edge(
            bond.GetBeginAtomIdx(),
            bond.GetEndAtomIdx(),
        )
    return g


def rdmolnxeq(m1, m2):
    g1 = rdmol2nxg(m1)
    g2 = rdmol2nxg(m2)
    return nx.is_isomorphic(g1, g2, node_match=iso.generic_node_match('symbol', None, eq))


def get_cutoffdict(oxide_structure: Structure, scale=1.3):
    cutoff_dict = dict()
    e_strings = list(set(s.species_string for s in oxide_structure))
    for pair in itertools.combinations(e_strings, 2):
        e1: Composition
        e1, e2 = pair
        cutoff_dict[(e1, e2)] = (AtomicRadius_string(e1) + AtomicRadius_string(e2)) * scale
    return cutoff_dict


from pymatgen.core.periodic_table import Element

_MetalDefined = {"Li", "Be", "Na", "Mg", "Al", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
                 "Ga",
                 "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Cs", "Ba", "La",
                 "Ce",
                 "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re",
                 "Os",
                 "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am",
                 "Cm",
                 "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh",
                 "Fl", "Mc", "Lv"}
_MetalloidDefined = {"Si", "Te", "Sb", "Ge"}

MDefined = set.union(_MetalDefined, _MetalloidDefined)

NMDefined = set([e for e in AllElements if e not in MDefined])

BuildingUnitElements = {"Si", "B", "C", "O", "H", "N", "F", "P", "S", "Cl", "As", "Se", "Br", "I", }


def is_metal_defined(e: Element or str):
    if isinstance(e, Element):
        e = e.symbol
    return e in _MetalDefined


def is_csd_metal(e: Element or str):
    if e == "D":
        e = "H"
    if isinstance(e, str):
        e = Element(e)
    if e.is_metal or e.symbol in ("Ge", "Sb", "Po"):
        return True
    return False


def is_halogen(e: Element or str):
    if e == "D":
        e = "H"
    if isinstance(e, str):
        e = Element(e)
    if e.is_halogen:
        return True
    return False


def graph_json_dump(g, jsonfile):
    mg_json_data = json_graph.node_link_data(g)
    with open(jsonfile, "w") as f:
        json.dump(mg_json_data, f)


def graph_json_dumps(g):
    mg_json_data = json_graph.node_link_data(g)
    return json.dumps(mg_json_data)


def graph_json_load(jsonfile):
    with open(jsonfile, "r") as f:
        mg_json_data = json.load(f)
    return json_graph.node_link_graph(mg_json_data)


def graph_json_loads(js):
    mg_json_data = json.loads(js)
    return json_graph.node_link_graph(mg_json_data)


from functools import wraps
import errno
import os
import signal


class TimeoutError(Exception):
    pass


def timeout(seconds=10, error_message=os.strerror(errno.ETIME)):
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutError(error_message)

        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result

        return wraps(func)(wrapper)

    return decorator


def get_mo_compositions_from_csd(csdformula: str):
    metaloxide = Composition()
    for c in csdformula2pmg_composition(csdformula, neutralize_hcations=True):
        c = get_no_hydrogen_composition(c)
        if any(m in c for m in MDefined) and "O" in c and c.num_atoms > 2:
            metaloxide += c
    return Composition(metaloxide.alphabetical_formula)


def get_mo_compositions_from_sslist(sslist: [Structure]):
    metaloxide = Composition()
    for s in sslist:
        c = get_no_hydrogen_composition(s.composition)
        if any(m in c for m in MDefined) and "O" in c and c.num_atoms > 2:
            metaloxide += c
    return Composition(metaloxide.alphabetical_formula)


def read_gcd(fn: typing.Union[str, pathlib.Path]):
    with open(fn, "r") as f:
        identifiers = [line.strip() for line in f]
    return identifiers


def comp_contains_elements(c: Composition, es: [str]):
    return bool(set(e.symbol for e in c.elements).issuperset(set(es)))


def comp_intersect_elements(c: Composition, es: [str]):
    return bool(set(e.symbol for e in c.elements).intersection(set(es)))


def comp_equals_elements(c: Composition, es: [str]):
    return set(e.symbol for e in c.elements) == set(es)


def available_values_by_field(df: pd.DataFrame, field: str, count_limit=0):
    vcdict = dict(df[field].value_counts())
    uniquecol_list = list(k for k in vcdict.keys() if vcdict[k] >= count_limit)
    return uniquecol_list


def count_rows_by_column1_by_unique_values_of_column2(df: pd.DataFrame, col1="dimension", col2="smiles", cl=0):
    unique_col1 = available_values_by_field(df, col1, count_limit=cl)
    unique_col2 = available_values_by_field(df, col2, count_limit=cl)
    nrs = []
    for ucol2 in unique_col2:
        total_counts = 0
        r = {col2: ucol2}
        for ucol1 in unique_col1:
            xd = df.loc[df[col2] == ucol2]
            count = len(xd.loc[xd[col1] == ucol1])
            total_counts += count
            r[ucol1] = count
        r["total"] = total_counts
        nrs.append(r)
    nrs = sorted(nrs, key=lambda x: x["total"], reverse=True)
    return pd.DataFrame.from_records(nrs)

def find_ms(symbols: [str]):
    ms = []
    for s in symbols:
        if s in MDefined:
            ms.append(s)
    return sorted(ms)


def find_nms(symbols: [str]):
    ms = []
    for s in symbols:
        if s not in MDefined:
            ms.append(s)
    return sorted(ms)

extra_smis = [
    "NC1CCNCC1",
    "CN(C)C(C)(C)CCN",
    "CCCCCCCCN1CCNCC1",
    "NCC1CCC(CN)CC1",
    "C[N+]12CN3CN(CN(C3)C1)C2",
]
