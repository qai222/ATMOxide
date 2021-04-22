import copy
import re
from collections import Counter
from io import StringIO
from operator import eq

import matplotlib.pylab as plt
import networkx as nx
import networkx.algorithms.isomorphism as iso
import numpy as np
from monty.json import MSONable
from pymatgen.core.periodic_table import Element
from pymatgen.vis.structure_vtk import EL_COLORS
from rdkit import Chem
from rdkit.Chem import rdmolops

from AnalysisModule.routines.conformer_parser import ACParser
from AnalysisModule.routines.util import AllElements, MDefined, graph_json_dumps, BuildingUnitElements
from AnalysisModule.routines.util import json_graph, Composition

plt.rcParams['svg.fonttype'] = 'none'


class ChemicalDiagram:

    def __init__(self, graph: nx.Graph):
        self.graph = graph
        symbols = nx.get_node_attributes(self.graph, "symbol")
        dnodes = [n for n in self.graph.nodes if symbols[n].upper() == "D"]
        nx.set_node_attributes(self.graph, {k: "H" for k in dnodes}, "symbol")
        self.symbols = nx.get_node_attributes(self.graph, "symbol")
        self.m_nodes = [k for k, v in self.symbols.items() if v in MDefined]

    @classmethod
    def from_entry(cls, entry):
        e = entry._entry
        cd = e.chemical_diagram()
        graph = nx.Graph()
        for i in range(cd.natoms()):
            a1 = cd.atom(i)
            side_label = a1.side_label()
            symbol = str(a1.centre_label())
            charge_label = str(a1.top_right_label())

            if len(charge_label) == 0:
                charge = 0
            else:
                try:
                    charge = int(re.findall(r"\d+", charge_label)[0])
                except IndexError:
                    charge = 1
                if "-" in charge_label:
                    charge = - charge
                elif "+" in charge_label:
                    pass
                else:
                    raise ValueError("charge_label is strange: {}".format(charge_label))
            iso_label = str(a1.top_left_label())
            side_element_label = str(side_label[0])
            try:
                side_element_count = int(side_label[1])
            except ValueError:
                if len(side_element_label) > 0:
                    side_element_count = 1
                else:
                    side_element_count = None
            if side_element_count is None:
                show_label = iso_label + symbol + charge_label + side_element_label
            else:
                show_label = iso_label + symbol + charge_label + side_element_label + str(side_element_count)
            attr = dict(
                symbol=symbol,
                charge_label=charge_label,
                iso_label=iso_label,
                side_element_label=side_element_label,
                side_element_count=side_element_count,
                show_label=show_label,
                charge=charge
            )

            try:
                pos = a1.site().position()
                graph.add_node(i, x=pos.x(), y=pos.y(), **attr)
            except AttributeError:
                graph.add_node(i, x=None, y=None, **attr)

        for i in range(cd.natoms()):
            for j in range(i + 1, cd.natoms()):
                if cd.bond_index(i, j) != -1:
                    graph.add_edge(i, j)
        return cls(graph)

    def check_symbols(self):
        if set(self.symbols.keys()).issubset(AllElements):
            return True
        return False

    def get_component_element_sets(self):
        element_sets = []
        for c in nx.connected_components(self.graph):
            element_sets.append(set([self.symbols[n] for n in c]))
        return element_sets

    def get_m_env(self):
        """env[metal_node]=a list of nb elements"""
        env = {}
        for mn in self.m_nodes:
            env[mn] = [self.symbols[nb] for nb in nx.neighbors(self.graph, mn)]
        return env

    def get_x_env(self, x: str):
        env = {}
        x_nodes = [k for k, v in self.symbols.items() if v == x]
        for mn in x_nodes:
            env[mn] = [self.symbols[nb] for nb in nx.neighbors(self.graph, mn)]
        return env

    def get_oxygens_nbing_m(self):
        oxygens_nbing_m = []
        for k, v in self.symbols.items():
            if v == "O" and set(nx.neighbors(self.graph, k)).intersection(set(self.m_nodes)):
                oxygens_nbing_m.append(k)
        return oxygens_nbing_m

    def is_nb2h(self, n):
        nbs = nx.neighbors(self.graph, n)
        nbs_symbols = [self.symbols[nb] for nb in nbs]
        if Counter(nbs_symbols)["H"] == 2:
            return True
        return False

    def is_nb1honly(self, n):
        nbs = nx.neighbors(self.graph, n)
        nbs_symbols = [self.symbols[nb] for nb in nbs]
        if Counter(nbs_symbols)["H"] == 1 and len(nbs_symbols) == 2:
            return True
        return False

    def has_subgraph(self, subgraph: nx.Graph):
        matcher = iso.GraphMatcher(self.graph, subgraph, node_match=iso.generic_node_match('symbol', None, eq))
        return matcher.subgraph_is_isomorphic()

    def all_aquo_complex(self):
        return all(self.is_nb2h(n) for n in self.get_oxygens_nbing_m())

    def all_hydroxide(self):
        return all(self.is_nb1honly(n) for n in self.get_oxygens_nbing_m())


class CDFilter:

    def __init__(self,
                 inclusion_subgraphs: [nx.Graph],
                 exclusion_subgraphs: [nx.Graph],
                 cd_filter_functions: list):

        self.inclusion_subgraphs = inclusion_subgraphs
        self.exclusion_subgraphs = exclusion_subgraphs
        self.cd_filter_functions = cd_filter_functions

    def accept(self, cd: ChemicalDiagram):
        logmsgs = []
        for filter_function in self.cd_filter_functions:
            if filter_function(cd):
                logmsgs.append("pass: {}".format(filter_function.__doc__))
            else:
                logmsgs.append("failed: {}".format(filter_function.__doc__))
                return False, "\n".join(logmsgs)
        for subg in self.inclusion_subgraphs:
            if cd.has_subgraph(subg):
                logmsgs.append("pass: {}".format(subg.graph["instruction"]))
            else:
                logmsgs.append("failed: {}".format(subg.graph["instruction"]))
                return False, "\n".join(logmsgs)
        for subg in self.exclusion_subgraphs:
            if cd.has_subgraph(subg):
                logmsgs.append("failed: {}".format(subg.graph["instruction"]))
                return False, "\n".join(logmsgs)
            else:
                logmsgs.append("pass: {}".format(subg.graph["instruction"]))
        return True, "\n".join(logmsgs)

    @staticmethod
    def gen_line_graph(atoms: [str]):
        g = nx.Graph()
        for i, s in enumerate(atoms):
            g.add_node(i, symbol=s)
        for i in range(len(atoms) - 1):
            g.add_edge(i, i + 1)
        return g

    @staticmethod
    def gen_nb_graph(center: str, nbs: [str]):
        g = nx.Graph()
        symbols = [center] + nbs
        for i, s in enumerate(symbols):
            g.add_node(i, symbol=s)
        for i in range(1, len(symbols)):
            g.add_edge(0, i)
        return g

    @staticmethod
    def parse_SubGraphRules(rules: str):
        in_subgs = []
        ex_subgs = []

        expand_M_rules = []
        rules = rules.split("\n")[1:-1]
        for rule in rules:
            if "M-" in rule:
                for m in MDefined:
                    expand_M_rules.append(rule.replace("M-", "{}-".format(m)))
        rules = rules + expand_M_rules
        for rule in rules:
            rule = rule.strip().strip(",")
            if "line graph" in rule:
                atoms = [w for w in rule.split() if "-" in w][-1].split("-")
                subg = CDFilter.gen_line_graph(atoms)
                subg.graph["instruction"] = rule
            elif "neighbor graph" in rule:
                center = [w for w in rule.split() if "*" in w][-1][1:]
                nbs = [w for w in rule.split() if ";" in w][-1].split(";")
                subg = CDFilter.gen_nb_graph(center, nbs)
                subg.graph["instruction"] = rule
            else:
                raise ValueError("rule not understood: {}".format(rule))
            if "inclusion" in rule:
                in_subgs.append(subg)
            elif "exclusion" in rule:
                ex_subgs.append(subg)
            else:
                raise ValueError("rule in/ex not understood: {}".format(rule))
        return in_subgs, ex_subgs


class CdBreakerError(Exception): pass


class CdBreaker:
    CNH_subgraph = CDFilter.gen_line_graph(["C", "N", "H"])

    @staticmethod
    def check_amine_like_mg(mg: nx.Graph):
        exception_msgs = []
        # check total charge
        total_charge = 0
        for n in mg.nodes(data=True):
            # check amid
            if n[1]["symbol"] == "N":
                neighbors = mg.neighbors(n[0])
                nb_nonhcount = 0
                nb_hcount = 0
                for nb in neighbors:
                    if mg.nodes[nb]["symbol"] != "H":
                        nb_nonhcount += 1
                    else:
                        nb_hcount += 1
                if n[1]["charge"] > nb_hcount:
                    emsg = "positive atomic charge with fewer protons??"
                    exception_msgs.append(emsg)
                # if nb_nonhcount >= 3:
                #     emsg = "an amide??"
                #     exception_msgs.append(emsg)

            total_charge += n[1]["charge"]
        if total_charge < 0:
            emsg = "an amine anion??"
            exception_msgs.append(emsg)
        return exception_msgs

    @staticmethod
    def neutralize_amine_cation_mg(mg: nx.Graph):
        h2beremoved = []
        for n in mg.nodes(data=True):
            # check amid
            charge = n[1]["charge"]
            if n[1]["symbol"] == "N" and charge > 0:
                hneighbors = []
                for nb in mg.neighbors(n[0]):
                    if mg.nodes[nb]["symbol"] in ("H", "D"):
                        hneighbors.append(nb)
                h2beremoved += hneighbors[:charge]
        for hnode in h2beremoved:
            mg.remove_node(hnode)
        return mg

    @staticmethod
    def neutralize_amine_rdmol(mol):
        for atom in mol.GetAtoms():
            chg = atom.GetFormalCharge()
            if chg == 0:
                continue
            if chg < 0:
                raise CdBreakerError("discard: negative formal charge on {}".format(atom.GetSymbol()))
            hcount = atom.GetTotalNumHs()
            if hcount - chg < 0:
                raise CdBreakerError("discard: too few hydrogens on {}".format(atom.GetSymbol()))
            atom.SetFormalCharge(0)
            atom.SetNumExplicitHs(hcount - chg)
            atom.UpdatePropertyCache()
        return mol

    @staticmethod
    def breakdown_cdg(cdg: nx.Graph):
        symbols = nx.get_node_attributes(cdg, "symbol")
        dnodes = [n for n in cdg.nodes if symbols[n].upper() == "D"]
        nx.set_node_attributes(cdg, {k: "H" for k in dnodes}, "symbol")

        amine_like_mgs = []  # composition CNH
        mo_containing_mgs = []  # has metal and oxygen, so counter ions like Na+ is not here
        other_mgs = []  # all other stuff

        def is_mg_amine(mg: nx.Graph):
            mg_symbols = list(nx.get_node_attributes(mg, "symbol").values())
            if set(mg_symbols) != {"C", "N", "H"}:
                return False
            emsgs = CdBreaker.check_amine_like_mg(mg)
            if len(emsgs) > 0:
                return False
            matcher = iso.GraphMatcher(mg, CdBreaker.CNH_subgraph,
                                       node_match=iso.generic_node_match('symbol', None, eq))
            if not matcher.subgraph_is_isomorphic():
                return False
            return True

        def is_mg_mo(mg: nx.Graph):
            mg_symbols = list(nx.get_node_attributes(mg, "symbol").values())
            if MDefined.intersection(set(mg_symbols)) and "O" in mg_symbols:
                return True
            return False

        for c in nx.connected_components(cdg):
            mg = cdg.subgraph(c).copy()
            if is_mg_amine(mg):
                amine_like_mgs.append(mg)
            elif is_mg_mo(mg):
                mo_containing_mgs.append(mg)
            else:
                mg_symbols = list(nx.get_node_attributes(mg, "symbol").values())
                matcher = iso.GraphMatcher(mg, CdBreaker.CNH_subgraph,
                                           node_match=iso.generic_node_match('symbol', None, eq))
                if matcher.subgraph_is_isomorphic():
                    raise CdBreakerError("discard: one other mg has C~N~H, e.g. Cl-CH2-NH2")
                if Counter(mg_symbols)["C"] > 7:
                    raise CdBreakerError("discard: one other mg has more than 7 carbon atoms")
                other_mgs.append(mg)

        if len(mo_containing_mgs) < 1:
            raise CdBreakerError("discard: too few MO components")
        if len(amine_like_mgs) < 1:
            raise CdBreakerError("discard: too few amine_like components")

        smis = []
        for amg in amine_like_mgs:
            neutralized = CdBreaker.neutralize_amine_cation_mg(amg)
            remaining_charge = 0
            for n in neutralized.nodes(data=True):
                remaining_charge += n[1]["charge"]
            mol, smiles = CdBreaker.mg2rdmol(neutralized)
            smiles = Chem.CanonSmiles(smiles)
            mol = Chem.MolFromSmiles(smiles)
            if rdmolops.GetFormalCharge(mol) > 0:  # this should not happen as charge is set to 0 in mg2rdmol, idk...
                if rdmolops.GetFormalCharge(mol) > remaining_charge:
                    raise CdBreakerError("discard: excess charge")
                mol = CdBreaker.neutralize_amine_rdmol(mol)
                smiles = Chem.CanonSmiles(Chem.MolToSmiles(mol))
            #     emsg = "discard: 2rdmol failed"
            #     raise CdgBreakdownError(emsg)
            smis.append(smiles)
        if len(set(smis)) != 1:
            raise CdBreakerError("discard: more than one unique amine smiles")
        return amine_like_mgs, mo_containing_mgs, other_mgs, smis

    @staticmethod
    def mg2rdmol(g: nx.Graph):
        nodes = sorted(list(g.nodes(data=True)), key=lambda x: x[0])
        atomidx2nodename = {}  # d[atom idx in the new rdmol] = original graph node
        nodename2atomidx = {}

        atom_number_list = []
        total_charge = 0
        for i in range(len(nodes)):
            graph_node = nodes[i][0]  # this could be different form i!
            symbol = nodes[i][1]['symbol']
            if symbol == "D":
                symbol = "H"
            z = Element(symbol).Z
            atom_number_list.append(z)
            atomidx2nodename[i] = graph_node
            nodename2atomidx[graph_node] = i

        adj = nx.convert.to_dict_of_dicts(g)
        new_ac = np.zeros((len(g.nodes), len(g.nodes))).astype(int)  # atomic connectivity
        for i in range(len(g.nodes)):
            for j in range(i + 1, len(g.nodes)):
                if atomidx2nodename[j] in adj[atomidx2nodename[i]].keys():
                    new_ac[i, j] = 1
                    new_ac[j, i] = 1

        ap = ACParser(sani=True, ac=new_ac, atomnumberlist=atom_number_list, charge=total_charge,
                      apriori_radicals=None, )
        mol, smiles = ap.parse(charged_fragments=True, force_single=False, expliciths=False)
        return mol, smiles

    @staticmethod
    def breakdown_as_jdict(cd: ChemicalDiagram):
        amine_like_mgs, mo_containing_mgs, other_mgs, smis = CdBreaker.breakdown_cdg(cd.graph)
        output = dict(
            almgs=[graph_json_dumps(g) for g in amine_like_mgs],
            momgs=[graph_json_dumps(g) for g in mo_containing_mgs],
            othermgs=[graph_json_dumps(g) for g in other_mgs],
            smis=smis)
        return output


class BuildingUnit(MSONable):
    def __init__(self, g: nx.Graph, buid=None, cdg=None):
        self.graph = g
        self.symbols = sorted(x if x != "D" else "H" for x in nx.get_node_attributes(self.graph, "symbol").values())
        self.composition = Composition(" ".join(self.symbols))
        try:
            self.buid = int(buid)
        except TypeError:
            self.buid = buid
        self.cdg = cdg

    def as_dict(self):
        assert isinstance(self.buid, int)
        assert self.cdg != None
        json_data = json_graph.node_link_data(self.graph)
        cdg_data = json_graph.node_link_data(self.cdg)
        return dict(graph_data=json_data, buid=self.buid, cdg_data=cdg_data)

    @classmethod
    def from_dict(cls, d):
        graph_data = d["graph_data"]
        cdg_data = d["cdg_data"]
        buid = d["buid"]
        g = json_graph.node_link_graph(graph_data)
        cdg = json_graph.node_link_graph(cdg_data)
        return cls(g, buid, cdg=cdg)

    # def as_json(self):
    #     assert isinstance(self.buid, int)
    #     assert self.cdg != None
    #     json_data = json_graph.node_link_data(self.graph)
    #     cdg_data = json_graph.node_link_data(self.cdg)
    #     return json.dumps(dict(graph_data=json_data, buid=self.buid, cdg_data=cdg_data))
    #
    # @classmethod
    # def from_json(cls, bujson):
    #     d = json.loads(bujson)
    #     graph_data = d["graph_data"]
    #     cdg_data = d["cdg_data"]
    #     buid = d["buid"]
    #     g = json_graph.node_link_graph(graph_data)
    #     cdg = json_graph.node_link_graph(cdg_data)
    #     return cls(g, buid, cdg=cdg)

    def contains_graph(self, subgraph: nx.Graph):
        matcher = iso.GraphMatcher(self.graph, subgraph, node_match=iso.generic_node_match('symbol', None, eq))
        return matcher.subgraph_is_isomorphic()

    def contains_bu(self, other):
        matcher = iso.GraphMatcher(self.graph, other.graph, node_match=iso.generic_node_match('symbol', None, eq))
        return matcher.subgraph_is_isomorphic()

    @property
    def is_allowed(self):
        return bool(set(nx.get_node_attributes(self.graph, "symbol").values()).issubset(BuildingUnitElements))

    @property
    def element_set(self):
        return set(self.symbols)

    def __len__(self):
        return len(self.graph.nodes)

    def __repr__(self):
        return "BUid-{}:/{}/".format(self.buid, self.composition.formula)

    def __hash__(self):
        return hash(self.composition)

    def __eq__(self, other):
        if self.composition != other.composition:
            return False
        g1 = self.graph
        g2 = other.graph
        return nx.is_isomorphic(g1, g2, node_match=iso.generic_node_match('symbol', None, eq))

    def draw_by_cdg(self, title="", urltxt=None, urllink=None):
        f = plt.figure()
        ax = plt.gca()
        ax.set_title(title)
        cdg = self.cdg
        posx = nx.get_node_attributes(cdg, 'x')
        posy = nx.get_node_attributes(cdg, 'y')
        cdg_labels = nx.get_node_attributes(cdg, 'show_label')
        cdg_symbols = nx.get_node_attributes(cdg, 'symbol')
        pltgraph = copy.deepcopy(self.graph)
        coords = {}
        subset_symbols = {}
        show_lables = {}
        missingxy = []
        for k in pltgraph.nodes:
            x = posx[k]
            y = posy[k]
            if x is None or y is None:
                missingxy.append(k)
                continue
            coords[k] = (posx[k], posy[k])
            show_lables[k] = cdg_labels[k]
            subset_symbols[k] = cdg_symbols[k]
        for k in missingxy:
            pltgraph.remove_node(k)
        jmolcolors = []
        for n in pltgraph.nodes:
            if subset_symbols[n] == "D":
                subset_symbols[n] = "H"
            jmolcolors.append('#{:02x}{:02x}{:02x}'.format(*EL_COLORS['Jmol'][subset_symbols[n]]))
        nx.draw(pltgraph, with_labels=True, labels=show_lables, pos=coords, ax=ax, node_color=jmolcolors)

        if urltxt and urllink:
            xycoords_array = [np.array(list(xy)) for xy in coords.values()]
            center = np.mean(xycoords_array, axis=0)
            x, y = center
            plt.text(x, y, urltxt, url=urllink, bbox=dict(alpha=0.4, url=urllink, facecolor="red"))

        imgdata = StringIO()
        f.savefig(imgdata, format='svg')
        imgdata.seek(0)  # rewind the data
        svg_dta = imgdata.read()  # this is svg data
        plt.close(f)
        return svg_dta

    @staticmethod
    def GetBuildingUnits(mo_graph: nx.Graph):
        """
        the building units are obtained from a graph component containing metal and oxygen, via the following workflow:
        1. O H X atoms are removed where X is an element that is not in BuildingUnit elements
        2. connected components now are choosen as BuildingUnitCore
        3. extend each of BuildingUnitCore to the first order neighbours (// we can limit such neighbours to be oxygen or hydrogen)
            resulting subgraph is the BuildingUnit, one for each BuildingUnitCore

        :param mo_graph:
        :return:
        """
        reduced_graph = mo_graph.copy()
        toberemoved = []
        symbol_dict = nx.get_node_attributes(mo_graph, "symbol")

        for node in reduced_graph.nodes:
            if symbol_dict[node] in ("O", "H", "D") or symbol_dict[node] not in BuildingUnitElements:
                toberemoved.append(node)

        for node in toberemoved:
            reduced_graph.remove_node(node)

        building_units_core = [reduced_graph.subgraph(c).copy() for c in nx.connected_components(reduced_graph)]
        bus = []
        for buc in building_units_core:
            building_unit = []
            for buc_node in buc.nodes:
                building_unit.append(buc_node)
                building_unit = building_unit + list(mo_graph.neighbors(buc_node))
            building_unit = list(set(building_unit))
            building_unit_graph = mo_graph.subgraph(building_unit).copy()
            bus.append(BuildingUnit(building_unit_graph, buid=None, cdg=None))
        bus = sorted(bus, key=lambda x: len(x), reverse=True)
        reduced_bus = []
        for bu in bus:
            rbu: BuildingUnit
            if bu.element_set not in [rbu.element_set for rbu in reduced_bus]:
                reduced_bus.append(bu)
            else:
                if any(rbu.contains_bu(bu) for rbu in reduced_bus) and len(
                        bu) <= 3:  # this is for those bonds across boxes
                    continue
                else:
                    reduced_bus.append(bu)
        return reduced_bus
