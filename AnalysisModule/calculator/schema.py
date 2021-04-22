import logging

from pymatgen.core.structure import Structure, Composition, PeriodicSite, Element

from AnalysisModule.routines.util import MDefined
from AnalysisModule.routines.util import strip_elements

logger = logging.getLogger(__name__)


class SchemaError(Exception): pass


class HybridOxide:

    def __init__(
            self,
            oxide_structures: [Structure],
            template_structures: [Structure],
            spacefill_structures: [Structure],
            identifier: str,
            details: dict = None,
    ):
        """
        a hybrid oxide structure consists of oxide and other stuff

        :param oxide_structures:
        :param template_structures:
        :param spacefill_structures:
        :param identifier: string
        """
        self.oxide_structures = oxide_structures
        self.template_structures = template_structures
        self.spacefill_structures = spacefill_structures

        all_sites = []
        for ss in self.template_structures + self.spacefill_structures + self.oxide_structures:
            all_sites += ss.sites
        self.total_structure = Structure.from_sites(all_sites)
        self.identifier = identifier
        if details is None:
            self.details = {}
        else:
            self.details = details

    # def from_clean_structure(self, clean_structure):
    @property
    def oxide_structure(self):
        oxsites = []
        for ss in self.oxide_structures:
            oxsites += ss.sites
        return Structure.from_sites(oxsites)

    def as_dict(self):
        return {
            "oxide_structures": [oxs.as_dict() for oxs in self.oxide_structures],
            "template_structures": [ts.as_dict() for ts in self.template_structures],
            "spacefill_structures": [sps.as_dict() for sps in self.spacefill_structures],
            "total_structure": self.total_structure.as_dict(),
            "identifier": self.identifier,
            "details": self.details
        }

    @classmethod
    def from_dict(cls, d):
        oxs = [Structure.from_dict(oxs) for oxs in d["oxide_structures"]]
        ts = [Structure.from_dict(ts) for ts in d["template_structures"]]
        ss = [Structure.from_dict(sps) for sps in d["spacefill_structures"]]
        i = d["identifier"]
        details = d["details"]
        return cls(oxs, ts, ss, i, details)

    @staticmethod
    def is_oxide_structure(s: Structure):
        gt3atoms = len(s) > 3  # idk about this...
        gt2oxygen = len([Element("O") in site.species.elements for site in s]) > 2
        m_with_mobond = []
        for site in s:
            if site.species_string in MDefined:
                # if site.species.contains_element_type("metal") or site.species.contains_element_type("metalloid"):
                if {"o"}.issubset(set(ninfo[1].lower() for ninfo in site.properties["nn"])):
                    m_with_mobond.append(site)
        hasmobond = len(m_with_mobond) > 0
        return gt3atoms and gt2oxygen and hasmobond

    @staticmethod
    def is_template_structure(s: Structure):
        """
        - more than 1 atoms
        - has c..n(..h) bond
        - has no metal
        - has no metalloid
        """
        gt1atoms = len(s) > 1
        hascnhbond = len([site for site in s if site.species_string == "N" and {"C", "H"}.issubset(
            set(ninfo[1] for ninfo in site.properties["nn"]))]) > 0
        has_no_metal = not any(site.composition.contains_element_type("metal") for site in s)
        has_no_metalloid = not any(site.composition.contains_element_type("metalloid") for site in s)
        return gt1atoms and hascnhbond and has_no_metal and has_no_metalloid

    @staticmethod
    def is_amine_like(s: Structure, known_smiles=False):
        gt1atoms = len(s) > 1
        cnh = set(s.composition.elements) == set(Composition("C N H").elements)
        cn = set(s.composition.elements) == set(Composition("C N").elements)
        # # this is not reliable as ghost H may not even in the nn info, e.g. EKIDEH
        # hascnhbond = len([site for site in s if site.species_string == "N" and {"C", "H"}.issubset(
        #     set(ninfo[1] for ninfo in site.properties["nn"]))]) > 0
        # return gt1atoms and (cnh or cn) and hascnhbond
        hascnbond = len([site for site in s if site.species_string == "N" and {"C"}.issubset(
            set(ninfo[1] for ninfo in site.properties["nn"]))]) > 0
        return gt1atoms and (cnh or cn) and hascnbond

    @staticmethod
    def contains_nonehydrogen(s: Structure):
        # # this is not reliable as ghost H may not even in the nn info, e.g. EKIDEH
        for site in s:
            if "H" in [ninfo[1] for ninfo in site.properties["nn"] if ninfo[2] is None]:
                return True
        return False

    @staticmethod
    def implicit_hydrogen_count(s: PeriodicSite):
        # # this is not reliable as ghost H may not even in the nn info, e.g. EKIDEH
        return len([ninfo for ninfo in s.properties["nn"] if ninfo[1] == "H" and ninfo[2] is None])

    @classmethod
    def from_substructures(cls, substructure_list: [Structure], identifier, details=None, strip_alkali=True,
                           known_smiles=False):
        mox_ss = []
        temp_ss = []
        sf_ss = []
        ss: Structure
        for ss in sorted(substructure_list, key=lambda x: len(x), reverse=True):
            if strip_alkali:
                nss = strip_elements(ss, ["K", "Na", "Rb", "Cs", "Fr"])
            else:
                nss = ss
            if HybridOxide.is_oxide_structure(nss):
                mox_ss.append(nss)
            elif HybridOxide.is_amine_like(nss, known_smiles):
                temp_ss.append(nss)
            else:
                sf_ss.append(nss)
        if len(mox_ss) < 1:
            raise SchemaError("no metal oxide structure!")
        if len(temp_ss) < 1:
            logger.warning("no template structure!")
        return cls(
            mox_ss, temp_ss, sf_ss, identifier, details
        )


class TemplateStructure: pass


class SpaceFillStructure: pass
