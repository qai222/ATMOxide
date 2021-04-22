import datetime
import json
import logging

from pymatgen.core.structure import Lattice

from AnalysisModule.prepare.disclean_function import *
from AnalysisModule.prepare.saentry import SaotoEntry
from AnalysisModule.routines.util import Composition
from AnalysisModule.routines.util import strip_elements

try:
    from ccdc.search import TextNumericSearch
    from ccdc.entry import Entry, Citation, Molecule, Crystal
    from ccdc.molecule import Bond, Atom
    from ccdc.io import EntryReader
    from ccdc.io import csd_version
except ImportError:
    raise ImportError('You need CSD API to use this model')

logger = logging.getLogger(__name__)
CSD_classes = (Entry, Molecule, Atom, Bond, Crystal, Entry.CrossReference, Citation)
CSD_entry_reader = EntryReader('CSD')


def entry_disordercheck(e: Entry):
    if not e.has_3d_structure:
        msg = "no 3d structure"
        logger.warning(msg)
        return msg

    occus = []
    for a in e.disordered_molecule.atoms:
        o = a.occupancy
        if isinstance(o, float):
            if o < 1:
                occus.append('lt1')
            elif o > 1:
                occus.append('gt1')
            else:
                occus.append('1')
        else:
            occus.append('{}-{}'.format(a.atomic_symbol, str(o)))
    oset = set(occus)

    if not oset.issubset({"lt1", "1", "H-None"}):
        msg = "oset is not allowed: {}".format(oset)
        logger.warning(msg)
        return msg

    if "lt1" in oset and not e.has_disorder:
        msg = "occus contradict csd disorder (has_disorder is False)!"
        logger.warning(msg)
        return msg

    if "lt1" not in oset and e.has_disorder:
        msg = "occus contradict csd disorder (has_disorder is True)!"
        logger.warning(msg)
        return msg
    return "normal"


def csd_versioncheck():
    return {
        'version': csd_version(),
        'number_of_entries': len(CSD_entry_reader)
    }


def as_dict_crossref(cr: Entry.CrossReference):
    d = {}
    d['type'] = cr.type
    d['text'] = cr.text
    d['identifiers'] = cr.identifiers
    return d


def Doi2CsdEntries(doi: str):
    """
    doi search, can be modified to other types of search defined in CSD API
    """
    text_numeric_search = TextNumericSearch()
    text_numeric_search.add_doi(doi)
    entries = []
    try:
        hits = text_numeric_search.search()
        if len(hits) == 0:
            logger.warning('no entry found for: {}'.format(doi))
        else:
            logger.info('found entries:', len(hits))
            for hit in hits:
                entries.append(hit.entry)
    except:
        logger.warning('search failed')
    return entries


def get_CsdEntry(identifier: str):
    return CSD_entry_reader.entry(identifier)


def CsdEntry2SaotoEntry(e: Entry, clean_structure: Structure = None,
                        source_object_name="disordered_molecule",
                        trust_csd_has_disorder=True,
                        disorder_clean_strategy="remove_labels_with_nonword_suffix",
                        ):
    d = {}
    d['cifstring'] = e.to_string('cif')
    attrnames = [a for a in dir(e) if not a.startswith('_')]
    for attrname in attrnames:
        attr = getattr(e, attrname)
        # print(attrname, type(attr))
        if not callable(attr):
            if isinstance(attr, tuple):
                if all(isinstance(item, Entry.CrossReference) for item in attr):
                    d[attrname] = [as_dict_crossref(item) for item in attr]
                    continue
                elif all(isinstance(item, Citation) for item in attr):
                    d[attrname] = [dict(item._asdict()) for item in attr]  # Citation is a named tuple
                    continue
            elif isinstance(attr, datetime.date):
                d[attrname] = attr.isoformat()
            elif type(attr) not in CSD_classes:
                d[attrname] = attr

    class ME(json.JSONEncoder):
        def default(self, o):
            return o.__repr__()

    j = json.dumps(d, cls=ME)
    d = json.loads(j)
    if not isinstance(clean_structure, Structure):
        clean_structure, disordered_structure, cleanupdetails = CsdDisorderParser(source_object_name,
                                                                                  trust_csd_has_disorder,
                                                                                  disorder_clean_strategy).get_clean_configuration(
            e)
        d["cleanup_details"] = cleanupdetails
    else:
        d["cleanup_details"] = None
        disordered_structure = None
    return SaotoEntry(identifier=e.identifier, details=d, clean_structure=clean_structure,
                      disordered_structure=disordered_structure)


class CsdDisorderParser:

    def __init__(
            self,
            source_object_name="disordered_molecule",
            trust_csd_has_disorder=True,
            disorder_clean_strategy="remove_labels_with_nonword_suffix",
    ):
        """
        parse a configuration of CSD Entry to a pmg Structure

        :param source_object_name: from where the atoms and their properties are obtained
            "disordered_molecule"
            "asymmetric_unit_molecule"
            "molecule"
        :param trust_csd_has_disorder: do we trust Entry.has_disorder field in CSD API?
            if we trust it, then for disorder-free entries the disorder clean step will be skiped
        :param disorder_clean_strategy: how should we get a sensible configuration from it?
            "remove_labels_with_nonword_suffix": remove atoms whose labels ending with a non-word tag
        """
        self.source_object_name = source_object_name
        self.trust_csd_has_disorder = trust_csd_has_disorder
        self.disorder_clean_strategy = disorder_clean_strategy

        logger.info("####### CEP starts")
        logger.info("source_object_name: {}".format(self.source_object_name))
        logger.info("trust_csd_has_disorder: {}".format(self.trust_csd_has_disorder))
        logger.info("disorder_clean_strategy: {}".format(self.disorder_clean_strategy))

    def get_disordered_structure(
            self, e: Entry,
            strip_alkali=False
    ):
        logger.info('### getting disordered structure:')
        logger.info('identifier: {}'.format(e.identifier))
        logger.info('disorder check: {}'.format(entry_disordercheck(e)))
        source_object = getattr(e.crystal, self.source_object_name)
        # self.logger.info('cif string: {}'.format(cifstring))
        lattice = Lattice.from_parameters(
            e.crystal._crystal.cell().a(),
            e.crystal._crystal.cell().b(),
            e.crystal._crystal.cell().c(),
            e.crystal._crystal.cell().alpha().degrees(),
            e.crystal._crystal.cell().beta().degrees(),
            e.crystal._crystal.cell().gamma().degrees(),
        )
        sites = []
        labels = []
        ghost_atom_lables = []
        icomp = 0
        logger.info('loop over components in the source object, start from the smaller one')
        for comp in sorted(source_object.components, key=lambda x: len(x.atoms)):
            logger.info('working on comp: {}'.format(comp.formula))
            for a in comp.atoms:
                try:
                    a_frac_coords = [float(xx) for xx in a.fractional_coordinates]
                except:
                    ghost_atom_lables.append(AtomLabel(a.label))
                    logger.warning(
                        'invalid coords found for atom: {}'.format(a.label))  # e.g. ghost None coords hydrogens
                    continue

                b: Bond
                if a.fractional_uncertainties is not None:
                    fu = [braket2float(str(fu)) for fu in a.fractional_uncertainties]
                else:
                    fu = None

                nn_infos = []
                for na in a.neighbours:
                    nn_info = (
                        na.label,
                        na.atomic_symbol,
                        csd_coords2list(na.fractional_coordinates),  # this could be None
                    )
                    if na.fractional_coordinates is None and na.atomic_symbol == 'H':
                        warnings.warn('there is a None coords hydrogen: {}'.format(na.label))
                    nn_infos.append(nn_info)

                # # this would break as there are bonds involving H-None atoms
                # for b in a.bonds:
                #     try:
                #         nn_info = (
                #             [na.label for na in b.atoms if na.label != a.label][0],
                #             [na.atomic_symbol for na in b.atoms if na.label != a.label][0], str(b.bond_type),
                #             b.ncrystals
                #         )
                #         nn_infos.append(nn_info)
                #     except:
                #         continue

                # this seems redundant, see csd_tests.properties.py
                try:
                    formal_charge = int(a.formal_charge)
                except:
                    logger.warning('formal charge is not an int!: {} -- {}'.format(a.label, a.formal_charge))
                    formal_charge = None

                prop = {
                    "formal_charge": formal_charge,
                    "label": a.label,
                    "occupancy": a.occupancy,
                    "fractional_uncertainty": fu,
                    "is_hb_acceptor": a.is_acceptor,
                    "is_hb_donor": a.is_donor,
                    "nn": nn_infos,
                    "icomp": icomp
                }
                # sites.append(PeriodicSite(a.atomic_symbol, a.coordinates, lattice, to_unit_cell=False, properties=prop,
                #                       coords_are_cartesian=True))  # this does NOT work properly
                if a.label not in labels:
                    sites.append(PeriodicSite(a.atomic_symbol, a_frac_coords, lattice, to_unit_cell=True,
                                              properties=prop,
                                              coords_are_cartesian=False))
                    labels.append(a.label)
                else:
                    labels.append(a.label)
                    logger.warning('duplicate labels in input, this is unlikely!: {}'.format(a.label))
            icomp += 1
        logger.info('grep symmetry operations from cif string...')
        cifstring = e.to_string('cif')
        ops = get_symmop_from_cif_string(cifstring)
        logger.info('# of symmops: {}'.format(len(ops)))
        logger.info('# of atoms in csd source object: {}'.format(len(source_object.atoms)))
        logger.info('# of sites before applying symm ops: {}'.format(len(sites)))

        psites = apply_symmop_merge_sites(sites, ops)
        # psites = apply_symmop_to_psites(sites, ops)  # assign field: iasym
        # psites = merge_psites(psites, sort_key="iasym", merge_key="label")
        logger.info('# of sites after applying symm ops: {}'.format(len(psites)))
        # logger.info('# of merged sites: {}'.format(len(sites) * len(ops) - len(psites)))
        disstructure = Structure.from_sites(psites)
        if strip_alkali:
            dstructure = strip_elements(disstructure, ["K", "Na", "Rb", "Cs", "Fr"])
            if disstructure.composition != dstructure.composition:
                logger.warning(
                    "alkali stripped!:\n\t{} --> {}".format(disstructure.composition, dstructure.composition))
        else:
            dstructure = disstructure
        logger.info('composition from csd entry: {}'.format(e.formula))
        logger.info('composition from csd {}: {}'.format(self.source_object_name, source_object.formula))
        logger.info('composition from resulting disordered structure: {}'.format(dstructure.composition))
        ghost_comp = Composition("")
        for ga in ghost_atom_lables:
            ghost_comp += Composition(ga.element)
        logger.info('ghost composition: {}'.format(ghost_comp))
        details = {
            '# of symmops': len(ops),
            '# of atoms in csd source object': len(source_object.atoms),
            '# of sites before applying symm ops': len(sites),
            '# of sites after applying symm ops': len(psites),
            'csd source object': self.source_object_name,
            'formula from csd entry': e.formula,
            'formula from csd source object': source_object.formula,
            'alkali strip': strip_alkali,
            'composition from resulting disordered structure': dstructure.composition.formula,
            'ghost composition': ghost_comp.formula,
            'disorder check': entry_disordercheck(e),
        }
        return dstructure, details

    def get_clean_configuration(self, e: Entry):
        disordered_structure, details = self.get_disordered_structure(e)
        logger.info('### cleanup disorder')
        logger.info('csd has_disorder: {}'.format(e.has_disorder))
        logger.info('clean method: {}'.format(self.disorder_clean_strategy))
        if self.trust_csd_has_disorder:
            logger.info('we trust csd has_disorder')
        else:
            logger.info('we DO NOT trust csd has_disorder')

        if self.trust_csd_has_disorder and not e.has_disorder or self.disorder_clean_strategy == "nothing":
            logger.info('so the input structure is returned as clean structure')
            clean_sites = disordered_structure.sites
        elif self.disorder_clean_strategy == "remove_labels_with_nonword_suffix":
            clean_sites = self.remove_sites_by_label_suffix(disordered_structure.sites)
        elif self.disorder_clean_strategy == "remove_labels_with_nonempty_suffix":
            clean_sites = self.remove_sites_by_label_suffix(disordered_structure.sites, keep_tag_regex=r"")
        else:
            raise CsdDiscleanError('clean method not implemented!')

        logger.info('# sites before clean: {}'.format(len(disordered_structure.sites)))
        logger.info('# sites after clean: {}'.format(len(clean_sites)))
        clean_structure = Structure.from_sites(clean_sites, to_unit_cell=True)
        logger.info('composition before clean: {}'.format(disordered_structure.composition.formula))
        logger.info('composition after clean: {}'.format(clean_structure.composition.formula))
        logger.info("again, entry formula: {}".format(e.formula))
        details['composition before clean'] = disordered_structure.composition.formula
        details['composition after clean'] = clean_structure.composition.formula
        return clean_structure, disordered_structure, details

    @staticmethod
    def remove_sites_by_label_suffix(sites, keep_tag_regex=r"^[A-z]{1,2}$", keep_empty_tag=True):
        def keep_tag(tag: str):
            if keep_empty_tag:
                return tag == "" or bool(re.match(keep_tag_regex, tag))
            else:
                bool(re.match(keep_tag_regex, tag))

        new_sites = []
        for s in sites:
            al = AtomLabel(s.properties['label'])
            if keep_tag(al.tag):
                new_sites.append(s)
                # new_sites.append(deepcopy(s))
        labels = [s.properties['label'] for s in new_sites]
        for s in new_sites:
            nns = s.properties['nn']
            new_nns = []
            for nn in nns:
                if nn[0] in labels:
                    new_nns.append(nn)
            s.properties['nn'] = new_nns
        return new_sites
