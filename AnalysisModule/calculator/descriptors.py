from copy import deepcopy

import pandas as pd
from matminer.featurizers import structure
from mordred import Calculator, descriptors
from pymatgen.analysis.graphs import ConnectedSite
from pymatgen.analysis.local_env import JmolNN, CrystalNN, CutOffDictNN
from pymatgen.core.structure import Structure, Site, Molecule, PeriodicSite
from rdkit import RDLogger
from rdkit.Chem import MolFromSmiles, Mol, Fragments

from AnalysisModule.calculator.cxcalc import JChemCalculator, Jia2019FeatureDict
from AnalysisModule.routines.pbc import PBCparser
from AnalysisModule.routines.util import get_cutoffdict

"""

classification of descriptors
- structures vs properties
- local vs global
- bulk vs organic  #?

iteractions:
- short contacts btw templates and inogranics
- dipole orientation?

total structure:
- density

template structure:
- smiles
- # of N
- charge
- packing?
- conformational?

inorganic structure:
- dimensionality
- metal coordination
- space fill
- conformational?
- MOF: LCD PLD
"""


class DescriptorError(Exception): pass


def is_smiles(s: str):
    try:
        RDLogger.DisableLog('rdApp.*')
        m = MolFromSmiles(s)
        RDLogger.EnableLog('rdApp.*')
        if isinstance(m, Mol):
            return True
        else:
            return False
    except:
        return False


class DescriptorCalculator:

    def __init__(self, entity):
        self.entity = entity

    def get_descriptors(self, updatedict=None, force_recal=False):
        if isinstance(updatedict, dict):
            ds = updatedict
        else:
            ds = dict()
        if force_recal:
            ds = dict()
        for cal in dir(self):
            if callable(getattr(self, cal)) and cal.startswith("cal_"):
                dname = cal[4:]
                if dname in ds.keys():
                    continue
                ds[dname] = getattr(self, cal)(self.entity)
        return ds


class MolecularDC(DescriptorCalculator):
    RdkitFrags = """_feat_fr_NH2 Number of primary amines
_feat_fr_NH1 Number of secondary amines
_feat_fr_NH0 Number of tertiary amines
_feat_fr_quatN Number of quaternary amines
_feat_fr_ArN Number of N functional groups attached to aromatics
_feat_fr_Ar_NH Number of aromatic amines
_feat_fr_Imine Number of imines
_feat_fr_amidine Number of amidine groups
_feat_fr_dihydropyridine Number of dihydropyridines
_feat_fr_guanido Number of guanidine groups
_feat_fr_piperdine Number of piperidine rings
_feat_fr_piperzine Number of piperzine rings
_feat_fr_pyridine Number of pyridine rings"""
    RdkitFragsFuncNames = [line.split()[0].replace("_feat_", "") for line in RdkitFrags.split("\n")]

    @staticmethod
    def cal_Mordred2D(smiles: str or [str]):
        mcalc = Calculator(descriptors, ignore_3D=True)
        if isinstance(smiles, str):
            mordred_des = mcalc(MolFromSmiles(smiles)).drop_missing().asdict()
        else:
            rdmols = [MolFromSmiles(smi) for smi in smiles]
            mordred_des = mcalc.pandas(rdmols)
        return mordred_des

    @staticmethod
    def cal_Jchem2D(smiles: str or [str]):
        jc = JChemCalculator()
        if isinstance(smiles, str):
            df = jc.cal_feature(list(Jia2019FeatureDict.values()), [smiles])
            del df['id']
            return df.to_dict("records")[0]
        else:
            df = jc.cal_feature(list(Jia2019FeatureDict.values()), smiles)
            del df['id']
            return df

    @staticmethod
    def cal_RdkitFrag(smiles: str or [str]):
        if isinstance(smiles, str):
            mol = MolFromSmiles(smiles)
            results = {}
            for funcname in MolecularDC.RdkitFragsFuncNames:
                if funcname in dir(Fragments):
                    func = getattr(Fragments, funcname)
                    if callable(func):
                        results[funcname] = func(mol)
            return results
        else:
            rdmols = [MolFromSmiles(smi) for smi in smiles]
            results = []
            for mol in rdmols:
                result = {}
                for funcname in MolecularDC.RdkitFragsFuncNames:
                    if funcname in dir(Fragments):
                        func = getattr(Fragments, funcname)
                        if callable(func):
                            result[funcname] = func(mol)
                results.append(result)
            return pd.DataFrame.from_records(results)


class ConformerDC(DescriptorCalculator): pass


class StructureDC(DescriptorCalculator):

    @staticmethod
    def s2c(anystructure: Structure, already_unwrap=True):
        if already_unwrap:
            unwrap_structure = anystructure
        else:
            mols, unwrap_structure, unwrap_pblock_list = PBCparser.unwrap(anystructure)
            if len(mols) > 1:
                raise DescriptorError("there are more than 1 molecule in the input of s2c!")
        s: PeriodicSite
        return Molecule.from_sites(
            Site(s.species, s.coords, properties=deepcopy(s.properties)) for s in unwrap_structure)

    @staticmethod
    def cal_LatticeVolume(anystructure: Structure):
        return anystructure.volume

    @staticmethod
    def cal_FormalCharge(anystructure: Structure):
        fc = 0
        for s in anystructure:
            try:
                fc += float(s.properties["formal_charge"])
            except:
                raise DescriptorError("formal charge is strange for site: {}".format(s.properties))
        return fc


class OxideStructureDC(StructureDC):

    @staticmethod
    def helper_DimensionPmg(oxide_structure: Structure):
        cutoff_dict = get_cutoffdict(oxide_structure, 1.3)
        dims = {}
        for nn in (
                # JmolNN(),
                # CovalentBondNN(),
                CrystalNN(),
                # VoronoiNN(),
                CutOffDictNN(cutoff_dict),
        ):
            dim = structure.Dimensionality(nn_method=nn)
            # if isinstance(nn, CrystalNN):
            #     bs = dim.nn_method.get_bonded_structure(oxide_structure)
            f = dim.featurize(oxide_structure)
            dims[nn.__class__.__name__] = int(f[0])
        return dims

    @staticmethod
    def helper_DimensionPmgWithoutHydrogens(oxide_structure: Structure):
        oxide_structure_woh = Structure.from_sites(
            [s for s in oxide_structure.sites if s.species_string.upper() != "H"])
        return OxideStructureDC.helper_DimensionPmg(oxide_structure_woh)

    @staticmethod
    def cal_Dimension(oxide_structure: Structure):
        dimension_dict = OxideStructureDC.helper_DimensionPmgWithoutHydrogens(oxide_structure)
        if len(set(dimension_dict.values())) != 1:
            raise DescriptorError("inconsistent dim calculated: {}".format(dimension_dict))
        else:
            # TODO add bias
            dim = list(dimension_dict.values())[0]
        return dim

    # @staticmethod
    # def cal_CoordinationDict(oxide_structure):
    #     possible_coordinations = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    #     coords = OrderedDict(zip(possible_coordinations, [[] for i in possible_coordinations]))
    #     for s in oxide_structure:
    #         coord = len(s.properties["nn"])
    #         if coord in possible_coordinations:
    #             coords[coord].append(s.properties["label"])
    #     return coords
    #
    # @staticmethod
    # def cal_CoordinationCounts(oxide_structure):
    #     table = OxideStructureDC.cal_CoordinationDict(oxide_structure)
    #     return [len(v) for v in table.values()]

    @staticmethod
    def cal_CompositionFormula(oxide_structure: Structure):
        return oxide_structure.composition.formula

    @staticmethod
    def helper_pbucenters(site: PeriodicSite):
        e = site.species.elements[0]
        if e.symbol == "H":
            return False
        if e.symbol == "O":
            return False
        if e.symbol == "F":
            return False
        if e.is_noble_gas:
            return False
        # if e.is_alkaline:
        #     return False
        if e.is_alkali:
            return False
        return True
        # if e.is_chalcogen or e.is_metal or e.is_metalloid:
        #     return True

    @staticmethod
    def helper_connectedsite2molsite(cs: ConnectedSite):
        ps: PeriodicSite = cs.site
        s = Site(ps.species, ps.coords, properties=ps.properties)
        return s
    #
    # @staticmethod
    # def cal_PrimaryBuildUnits(oxide_structure: Structure):
    #
    #     # TODO atomic radii bad, try cation/anion radii with CutoffNN
    #     # for nn in (
    #     #         JmolNN(),
    #     #         CrystalNN(),
    #     #         CutOffDictNN(cutoff_dict),
    #     # ):
    #     oxide_structure_woh = Structure.from_sites(
    #         [s for s in oxide_structure.sites if s.species_string.upper() != "H"])
    #     # cutoff_dict = OxideStructureDC.helpler_cutoffdict(oxide_structure_woh)
    #     # nn = CutOffDictNN(cutoff_dict)
    #     nn = JmolNN()
    #     dim = structure.Dimensionality(nn_method=nn)
    #     bs = dim.nn_method.get_bonded_structure(oxide_structure_woh)
    #     pbus = []
    #     for i in range(len(oxide_structure_woh.sites)):
    #         the_site: PeriodicSite = oxide_structure_woh.sites[i]
    #         if not OxideStructureDC.helper_pbucenters(the_site):
    #             continue
    #         # its_nbrs = getattr(bs.get_connected_sites(i), "ConnectedSite")
    #         pbu = the_site.species
    #         for cs in bs.get_connected_sites(i):
    #             pbu += Composition(cs.site.species_string)
    #         # its_nbrs = [cs.site for cs in bs.get_connected_sites(i)]  # TODO use helper_connectedsite2molsite to get a Molecule
    #         pbus.append(pbu.formula)
    #     return dict(Counter(pbus))
