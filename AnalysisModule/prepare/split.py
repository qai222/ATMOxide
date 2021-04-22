import logging
import warnings
from itertools import groupby

import networkx as nx
import numpy as np
from pymatgen.analysis.local_env import CrystalNN, JmolNN, ValenceIonicRadiusEvaluator, CutOffDictNN

from AnalysisModule.routines.pbc import PBCparser, Structure
from AnalysisModule.routines.util import get_cutoffdict

logger = logging.getLogger(__name__)

"""
- multiple NN algo can be used for establishing a bonded structure in `pymatgen`,
usually one needs `ValenceIonicRadiusEvaluator` to assign valences.
- `CrystalNN` does not work well with organics with atomic valences, will break a charged molecule.
- vanilla 1.3*covrad works ok for `norquist_sent` and `noquist_ppt`.
"""


class SplitError(Exception): pass


class Splitter:

    @staticmethod
    def split_structure_ocelot(s, cifname: str = None):
        logger.info('### SPLIT STRUCTURE -- ocelot')
        mols, structure, unwrap_pblock_list = PBCparser.unwrap_and_squeeze(s)  # assign imol site property
        substructures = []
        for imol, group in groupby(structure.sites, lambda x: x.properties['imol']):
            this_mol = [ps for ps in group]
            substructure = Structure.from_sites(this_mol)
            logger.info('substructure comp: {}'.format(substructure.composition))
            logger.info(
                'substructure reduced comp: {}'.format(substructure.composition.get_reduced_composition_and_factor()))
            try:
                icomp_set = set(ps.properties['icomp'] for ps in this_mol)
                if len(icomp_set) > 1:
                    logger.warning('icomp not consistent in an imol!: imol-{} icomps-{}'.format(imol, icomp_set))
            except KeyError:
                logger.warning('icomp not defined for all sites, skip icomp check')
            substructures.append(substructure)
        ss = sorted(substructures, key=lambda x: len(x), reverse=True)
        if cifname:
            sscifs = "\n".join([ss.to("cif") for ss in [structure] + ss])
            with open(cifname, "w") as f:
                f.write(sscifs)
        return structure, ss

    @staticmethod
    def split_structure_csdnn(s: Structure, hardcutoff=4):
        for site in s:
            if 'nn' not in site.properties:
                raise SplitError('nninfos not all assigned, cannot use csdnn to split!')
            nninfos = site.properties['nn']
            for info in nninfos:
                if info[2] is None:
                    raise SplitError('None coord in nninfos, please do addH first!')

        psites = s.sites
        if all('iasym' in site.properties for site in psites):
            sameiasym = True
            warnings.warn('if a percolating step involves hydrogens, only percolate to sites with the same iasym')
        else:
            sameiasym = False
        pindices = range(len(psites))
        visited = []
        unwrap_pblock_list = []
        while len(visited) != len(psites):
            # initialization
            unvisited = [idx for idx in pindices if idx not in visited]
            ini_idx = unvisited[0]
            block = [ini_idx]
            # unwrap.append(psites[ini_idx])
            unwrap_pblock = [psites[ini_idx]]
            pointer = 0
            while pointer != len(block):
                outside = [idx for idx in pindices if idx not in block and idx not in visited]
                for i in outside:
                    si = psites[i]
                    sj = psites[block[pointer]]
                    if sameiasym and (si.species_string == 'H' or sj.species_string == 'H'):
                        if si.properties['iasym'] != sj.properties['iasym']:
                            continue
                    distance, fctrans = PBCparser.get_dist_and_trans(s.lattice,
                                                                     psites[block[pointer]].frac_coords,
                                                                     psites[i].frac_coords, )
                    fctrans = np.rint(fctrans)
                    if psites[i].properties['label'] in [nninfo[0] for nninfo in psites[block[pointer]].properties[
                        'nn']] and distance < hardcutoff:
                        # if psites[block[pointer]].properties['label'] in [nninfo[0] for nninfo in psites[i].properties['nn']] and distance < hardcutoff:
                        block.append(i)
                        psites[i]._frac_coords += fctrans
                        unwrap_pblock.append(psites[i])
                visited.append(block[pointer])
                pointer += 1
            unwrap_pblock_list.append(unwrap_pblock)

        unwrap = []
        substructures = []
        for i in range(len(unwrap_pblock_list)):
            substructure = []
            for j in range(len(unwrap_pblock_list[i])):
                unwrap_pblock_list[i][j].properties['imol_csd'] = i
                unwrap.append(unwrap_pblock_list[i][j])
                substructure.append(unwrap_pblock_list[i][j])
            substructures.append(Structure.from_sites(substructure))
        return Structure.from_sites(unwrap), sorted(substructures, key=lambda x: len(x), reverse=True)

    # @staticmethod
    # def check_against_entry_formula(substructures: [Structure], eformula: str, stripalkli=True):
    #     logger.info('# check against entry formula')
    #     if stripalkli:
    #         eformula = strip_elements_from_formula(eformula, etype="alkali")
    #     csd_reduced_comp_set = set([c.reduced_composition for c in csdformula2pmg_composition(eformula)])
    #     ss: Structure
    #     split_reduced_comp_set = set([ss.composition.reduced_composition for ss in substructures])
    #
    #     exact_match = csd_reduced_comp_set == split_reduced_comp_set
    #     logger.info('e formula: {}'.format(eformula))
    #     logger.info("csd composition set: {}".format(csd_reduced_comp_set))
    #     logger.info("split composition set: {}".format(split_reduced_comp_set))
    #     logger.info("exact match: {}".format(exact_match))
    #
    #     csd_reduced_comp_set_no_hydrogen = []
    #     csdcomps = csdformula2pmg_composition(eformula)
    #     for c in csdcomps:
    #         nhc = get_no_hydrogen_composition(c)
    #         if nhc.num_atoms == 0:
    #             csd_reduced_comp_set_no_hydrogen.append(nhc)
    #         else:
    #             csd_reduced_comp_set_no_hydrogen.append(nhc.reduced_composition)
    #     csd_reduced_comp_set_no_hydrogen = set(csd_reduced_comp_set_no_hydrogen)
    #
    #     split_reduced_comp_set_no_hydrogen = []
    #     for ss in substructures:
    #         nhc = get_no_hydrogen_composition(ss.composition)
    #         if nhc.num_atoms == 0:
    #             split_reduced_comp_set_no_hydrogen.append(nhc)
    #         else:
    #             split_reduced_comp_set_no_hydrogen.append(nhc.reduced_composition)
    #     split_reduced_comp_set_no_hydrogen = set(split_reduced_comp_set_no_hydrogen)
    #     # csd_reduced_comp_set_no_hydrogen = set(
    #     #     get_no_hydrogen_composition(c).reduced_composition for c in csdformula2pmg_composition(eformula))
    #     # split_reduced_comp_set_no_hydrogen = set(
    #     #     get_no_hydrogen_composition(ss.composition).reduced_composition for ss in substructures)
    #     noh_match = csd_reduced_comp_set_no_hydrogen == split_reduced_comp_set_no_hydrogen
    #     logger.info("csd no-hydrogen composition set: {}".format(csd_reduced_comp_set_no_hydrogen))
    #     logger.info("split no-hydrogen composition set: {}".format(split_reduced_comp_set_no_hydrogen))
    #     logger.info("no hydrogen match: {}".format(noh_match))
    #
    #     comp: Composition
    #     largest_m_comp_csd = None
    #     for comp in sorted(list(csd_reduced_comp_set_no_hydrogen), key=lambda x: x.num_atoms, reverse=True):
    #         if comp_intersect_elements(comp, MDefined):
    #             largest_m_comp_csd = comp
    #             break
    #     if largest_m_comp_csd is None:
    #         raise SplitError("no M-containing component in csd comp set found!")
    #     largest_m_comp_split = None
    #     for comp in sorted(list(split_reduced_comp_set_no_hydrogen), key=lambda x: x.num_atoms, reverse=True):
    #         if comp_intersect_elements(comp, MDefined):
    #             largest_m_comp_split = comp
    #             break
    #     if largest_m_comp_split is None:
    #         raise SplitError("no M-containing component in split comp set found!")
    #
    #     lm_match = largest_m_comp_split == largest_m_comp_csd
    #     logger.info("csd largest M-containing composition set: {}".format(largest_m_comp_csd))
    #     logger.info("split largest M-containing composition set: {}".format(largest_m_comp_split))
    #     logger.info("lm match: {}".format(lm_match))
    #     return exact_match, noh_match, lm_match

    @staticmethod
    def split_structure_pymatgen(s: Structure, nn_model='cutoff', cifname: str = None):
        lattice = s.lattice
        if nn_model == 'jmol':
            nn = JmolNN()
        elif nn_model == 'cutoff':
            cutoffdict = get_cutoffdict(s, 1.3)
            nn = CutOffDictNN(cutoffdict)
        elif nn_model == 'crystal':
            # this method tends to break organics based on ValenceIonicRadiusEvaluator
            vire = ValenceIonicRadiusEvaluator(s)
            s = vire.structure
            nn = CrystalNN()
        else:
            raise SplitError("nn model not implemented: {}".format(nn_model))
        sg = nn.get_bonded_structure(s)
        supercell_sg = sg
        # supercell_sg = sg * (1, 1, 1)
        supercell_sg.graph = nx.Graph(supercell_sg.graph)

        # find subgraphs
        all_subgraphs = [supercell_sg.graph.subgraph(c) for c in
                         nx.connected_components(supercell_sg.graph)]

        # discount subgraphs that lie across *supercell* boundaries
        # these will subgraphs representing crystals
        molecule_subgraphs = []
        for subgraph in all_subgraphs:
            molecule_subgraphs.append(nx.MultiDiGraph(subgraph))

        # add specie names to graph to be able to test for isomorphism
        for subgraph in molecule_subgraphs:
            for n in subgraph:
                subgraph.add_node(n, specie=str(supercell_sg.structure[n].specie))

        ss = []
        for subgraph in molecule_subgraphs:
            coords = [supercell_sg.structure[n].coords for n
                      in subgraph.nodes()]
            species = [supercell_sg.structure[n].specie for n
                       in subgraph.nodes()]
            subgraph_structure = Structure(lattice, species, coords, coords_are_cartesian=True)
            ss.append(subgraph_structure)
        if cifname:
            sscifs = "\n".join([ss.to("cif") for ss in [s] + ss])
            with open(cifname, "w") as f:
                f.write(sscifs)

        return s, sorted(ss, key=lambda x: len(x), reverse=True)
