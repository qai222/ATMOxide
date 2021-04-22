import itertools
import re
import warnings
from copy import deepcopy

from pymatgen.core.operations import SymmOp
from pymatgen.core.structure import PeriodicSite, np, Structure
from pymatgen.io.cif import CifFile
from pymatgen.io.cif import _get_cod_data
from pymatgen.io.cif import sub_spgrp
from pymatgen.symmetry.groups import SYMM_DATA
from pymatgen.symmetry.groups import SpaceGroup
from pymatgen.util.coord import pbc_shortest_vectors


def csd_coords2list(coords):
    try:
        a_frac_coords = [float(xx) for xx in coords]
        return a_frac_coords
    except:
        return None


latt_labels = [
    '_cell_length_a', '_cell_length_b', '_cell_length_c', '_cell_angle_alpha', '_cell_angle_beta', '_cell_angle_gamma',
]

space_groups = {sub_spgrp(k): k for k in SYMM_DATA['space_group_encoding'].keys()}
space_groups.update({sub_spgrp(k): k for k in SYMM_DATA['space_group_encoding'].keys()})


class CsdDiscleanError(Exception): pass


def str2float(text):
    """
    Remove uncertainty brackets from strings and return the float.
    """

    try:
        # Note that the ending ) is sometimes missing. That is why the code has
        # been modified to treat it as optional. Same logic applies to lists.
        return float(re.sub(r"\(.+\)*", "", text))
    except TypeError:
        if isinstance(text, list) and len(text) == 1:
            return float(re.sub(r"\(.+\)*", "", text[0]))
    except ValueError as ex:
        if text.strip() == ".":
            return 0
        raise ex


def braket2float(s):
    try:
        return float(s)
    except ValueError:
        if isinstance(s, str):
            return str2float(s)
        raise TypeError('cannot parse {} into float'.format(s))


class AtomLabel:
    def __init__(self, label: str):
        """
        a class for atom label in cif file, overkill I guess...

        :param label:
        """
        self.label = label
        tmplabel = self.label
        self.element = re.findall(r"^[a-zA-Z]+", tmplabel)[0]

        tmplabel = tmplabel.lstrip(self.element)
        try:
            self.index = re.findall(r"^\d+", tmplabel)[0]
        except IndexError:
            self.index = "0"

        tmplabel = tmplabel.lstrip(str(self.index))
        self.tag = tmplabel

        # if len(self.tag) > 1:
        #     raise ValueError('tag for {} is {}, this is unlikely'.format(label, self.tag))

        self.index = int(self.index)

        self.ei = "{}{}".format(self.element, self.index)

    def __str__(self):
        return self.label

    def __repr__(self):
        return self.label

    def __hash__(self):
        return hash(self.label)

    def __eq__(self, other):
        return self.label == other.label

    @staticmethod
    def get_labels_with_tag(tag, als):
        sametag = []
        for alj in als:
            if tag == alj.tag:
                sametag.append(alj)
        return sametag

    def get_labels_with_same_ei(self, als):
        """
        get a list of atomlabel whose ei == al.ei

        :param als:
        :return:
        """
        sameei = []
        for alj in als:
            if self.ei == alj.ei and self != alj:
                sameei.append(alj)
        return sameei

    @staticmethod
    def get_psite_by_atomlable(psites: [PeriodicSite], al):
        for s in psites:
            if s.properties['label'] == str(al):
                return s
        raise ValueError('cannot find psite with atomlable {}'.format(str(al)))

    @classmethod
    def from_psite(cls, s: PeriodicSite):
        return cls(s.properties['label'])


def pbc_dist(fc1, fc2, lattice):
    v, d2 = pbc_shortest_vectors(lattice, fc1, fc2, return_d2=True)
    return np.sqrt(d2[0, 0])


def pbc_dist_array(fc1, fc2, lattice):
    v, d2 = pbc_shortest_vectors(lattice, fc1, fc2, return_d2=True)
    return np.sqrt(d2)


def apply_symmop_prune_op(psites, ops):
    """
    symmop and xyz in cif file:

    lets say xyz -- op1 --> x'y'z' and xyz -- op2 --> x!y!z! and
    it is possible to have x'y'z' is_close x!y!z!

    this means one should take only x'y'z' or x!y!z!, aka op1 is equivalent to op2 due to the symmetry implicated by
    xyz/the asymmectric unit, e.g. ALOVOO.cif -- Z=2, asymmectric unit given by cif is one molecule, but there're 4 ops

    so we need first check if the cif file behaves like this
    """
    op_xyzs = []
    for op in ops:
        n_xyzs = []
        for ps in psites:
            new_coord = op.operate(ps.frac_coords)
            # new_coord = np.array([i - math.floor(i) for i in new_coord])
            n_xyzs.append(new_coord)
        op_xyzs.append(n_xyzs)

    latt = psites[0].lattice

    def pbc_distmat(fcl1, fcl2):
        distmat = np.zeros((len(fcl1), len(fcl1)))
        for i in range(len(fcl1)):
            for j in range(i, len(fcl1)):
                distmat[i][j] = pbc_dist(fcl1[i], fcl2[j], latt)
                distmat[j][i] = distmat[i][j]
        return distmat

    def two_xyzs_close(xyzs1, xyzs2, tol=1e-5):
        dmat = pbc_distmat(xyzs1, xyzs2)
        almost_zeros = dmat[(dmat < tol)]
        if len(almost_zeros) > 0:
            return True
        return False

    op_identities = np.zeros((len(ops), len(ops)), dtype=bool)
    for i, j in itertools.combinations(range(len(ops)), 2):
        ixyzs = op_xyzs[i]
        jxyzs = op_xyzs[j]
        if two_xyzs_close(ixyzs, jxyzs):
            op_identities[i][j] = True
            op_identities[j][i] = True

    groups = [[0]]
    for i in range(len(ops)):
        for ig in range(len(groups)):
            if all(op_identities[i][j] for j in groups[ig]):
                groups[ig].append(i)
        if i not in [item for sublist in groups for item in sublist]:
            groups.append([i])
    unique_ops = [ops[g[0]] for g in groups]

    new_psites = apply_symmop_to_psites(psites, unique_ops)

    return new_psites, unique_ops


def merge_psites(psites, sort_key="iasym", merge_key="label", tol=1e-1):
    def same_merge_key(ps1, ps2, mkey):
        if mkey is None:
            return True
        else:
            return ps1.properties[mkey] == ps2.properties[mkey]

    psites_sorted = sorted(psites, key=lambda x: x.properties[sort_key])
    psites_sorted_coords = [ps.frac_coords for ps in psites_sorted]
    latt = psites_sorted[0].lattice
    distance_mat = pbc_dist_array(fc1=psites_sorted_coords, fc2=psites_sorted_coords, lattice=latt)
    distance_mat = np.array(distance_mat)
    pis, pjs = np.where(distance_mat < tol)
    # print("pis pjs", pis, pjs)
    conflicts = []
    for i in range(len(pis)):
        if pis[i] == pjs[i]:
            continue
        conflicts.append(pis[i])
        conflicts.append(pjs[i])
    pairs = dict()
    for i in range(len(pis)):
        pi = pis[i]
        pj = pjs[i]
        if pi < pj:
            # psi = psites_sorted[pi]
            # psj = psites_sorted[pj]
            # if same_merge_key(psi, psj, merge_key):
            #     warnings.warn(
            #         'the following two sites with the same {} are merged to the former: \n{}\n{}\n\n{}\n{}'.format(
            #             merge_key,
            #             psi,
            #             psi.properties,
            #             psj,
            #             psj.properties))
            # else:
            #     warnings.warn(
            #         'the following two sites with DIFFERENT {} are merged to the former: \n{}\n{}\n\n{}\n{}'.format(
            #             merge_key,
            #             psi,
            #             psi.properties,
            #             psj,
            #             psj.properties))
            pairs[pi] = pj
    new_psites = []
    for i in range(len(psites_sorted)):
        if i in pairs.keys():
            new_psites.append(psites_sorted[i])
        elif i not in set(conflicts):
            new_psites.append(psites_sorted[i])
    return new_psites


def apply_symmop_to_psites(psites: [PeriodicSite], ops):
    new_psites = []
    for ps in psites:
        iasym = 0
        for op in ops:
            new_coord = op.operate(ps.frac_coords)
            # new_coord = np.array([i - math.floor(i) for i in new_coord])
            # new_properties = deepcopy(ps.properties)
            new_properties = ps.properties
            new_properties['iasym'] = iasym
            new_ps = PeriodicSite(ps.species_string, new_coord, ps.lattice, properties=new_properties)
            new_psites.append(new_ps)
            iasym += 1
    return new_psites


def merge_sites_in_structure(structure: Structure, tol: float):
    # vanilla structure.merge_sites()
    from scipy.spatial.distance import squareform
    from scipy.cluster.hierarchy import fcluster, linkage

    d = structure.distance_matrix
    np.fill_diagonal(d, 0)
    clusters = fcluster(linkage(squareform((d + d.T) / 2)),
                        tol, 'distance')
    sites = []
    for c in np.unique(clusters):
        inds = np.where(clusters == c)[0]
        species = structure[inds[0]].species
        coords = structure[inds[0]].frac_coords
        props = structure[inds[0]].properties
        sites.append(PeriodicSite(species, coords, structure.lattice, properties=props))
    structure._sites = sites


def apply_symmop_merge_sites(psites: [PeriodicSite], ops: [SymmOp], tol=1e-1):
    """
    apply all ops and merge those within a tol
    """
    asym_structure = Structure.from_sites(psites)
    latt = asym_structure.lattice
    merge_sites_in_structure(asym_structure, tol)
    asym_structure.merge_sites(tol=tol, mode="delete")
    nsites = len(asym_structure)

    # mem save version
    sites = []
    op: SymmOp
    iasym = 0
    for op in ops:
        # print(iasym, len(sites))
        frac_coords = op.operate_multi(asym_structure.frac_coords)
        species = asym_structure.species
        props = deepcopy(asym_structure.site_properties)
        props["iasym"] = [iasym] * nsites
        opsites = [
            PeriodicSite(species[i], frac_coords[i], lattice=latt, properties=dict((k, v[i]) for k, v in props.items()))
            for i in range(nsites)]
        sites += opsites
        unit_cell = Structure.from_sites(sites)
        merge_sites_in_structure(unit_cell, tol)
        sites = unit_cell.sites
        iasym += 1
    unit_cell = Structure.from_sites(sites)
    return unit_cell.sites

    # sites = []
    # op: SymmOp
    # iasym = 0
    # for op in ops:
    #     frac_coords = op.operate_multi(asym_structure.frac_coords)
    #     species = asym_structure.species
    #     props = deepcopy(asym_structure.site_properties)
    #     props["iasym"] = [iasym] * nsites
    #     opsites = [PeriodicSite(species[i], frac_coords[i], lattice=latt, properties=dict((k, v[i]) for k,v in props.items())) for i in range(nsites)]
    #     sites += opsites
    #     iasym += 1
    # unit_cell = Structure.from_sites(sites)
    # merge_sites_in_structure(unit_cell, tol)
    # return unit_cell.sites


def get_symmop(data):
    symops = []
    for symmetry_label in ["_symmetry_equiv_pos_as_xyz",
                           "_symmetry_equiv_pos_as_xyz_",
                           "_space_group_symop_operation_xyz",
                           "_space_group_symop_operation_xyz_"]:
        if data.get(symmetry_label):
            xyz = data.get(symmetry_label)
            if isinstance(xyz, str):
                msg = "A 1-line symmetry op P1 CIF is detected!"
                warnings.warn(msg)
                xyz = [xyz]
            try:
                symops = [SymmOp.from_xyz_string(s)
                          for s in xyz]
                break
            except ValueError:
                continue
    if not symops:
        # Try to parse symbol
        for symmetry_label in ["_symmetry_space_group_name_H-M",
                               "_symmetry_space_group_name_H_M",
                               "_symmetry_space_group_name_H-M_",
                               "_symmetry_space_group_name_H_M_",
                               "_space_group_name_Hall",
                               "_space_group_name_Hall_",
                               "_space_group_name_H-M_alt",
                               "_space_group_name_H-M_alt_",
                               "_symmetry_space_group_name_hall",
                               "_symmetry_space_group_name_hall_",
                               "_symmetry_space_group_name_h-m",
                               "_symmetry_space_group_name_h-m_"]:
            sg = data.get(symmetry_label)

            if sg:
                sg = sub_spgrp(sg)
                try:
                    spg = space_groups.get(sg)
                    if spg:
                        symops = SpaceGroup(spg).symmetry_ops
                        msg = "No _symmetry_equiv_pos_as_xyz type key found. " \
                              "Spacegroup from %s used." % symmetry_label
                        warnings.warn(msg)
                        break
                except ValueError:
                    # Ignore any errors
                    pass

                try:
                    for d in _get_cod_data():
                        if sg == re.sub(r"\s+", "",
                                        d["hermann_mauguin"]):
                            xyz = d["symops"]
                            symops = [SymmOp.from_xyz_string(s)
                                      for s in xyz]
                            msg = "No _symmetry_equiv_pos_as_xyz type key found. " \
                                  "Spacegroup from %s used." % symmetry_label
                            warnings.warn(msg)
                            break
                except Exception:
                    continue

                if symops:
                    break
    if not symops:
        # Try to parse International number
        for symmetry_label in ["_space_group_IT_number",
                               "_space_group_IT_number_",
                               "_symmetry_Int_Tables_number",
                               "_symmetry_Int_Tables_number_"]:
            if data.get(symmetry_label):
                try:
                    i = int(braket2float(data.get(symmetry_label)))
                    symops = SpaceGroup.from_int_number(i).symmetry_ops
                    break
                except ValueError:
                    continue

    if not symops:
        msg = "No _symmetry_equiv_pos_as_xyz type key found. " \
              "Defaulting to P1."
        warnings.warn(msg)
        symops = [SymmOp.from_xyz_string(s) for s in ['x', 'y', 'z']]

    return symops


def cifstring2cifdata(cifstring):
    cifstring = cifstring
    cifdata = CifFile.from_string(cifstring).data
    identifiers = list(cifdata.keys())
    if len(identifiers) > 1:
        warnings.warn("find more than 1 structures in this cif file, only the first one is used!")
    elif len(identifiers) == 0:
        raise CsdDiscleanError("no structure found by pmg in cif string!")
    pymatgen_dict = list(cifdata.items())[0][1].data
    # jmol writes '_atom_site_type_symbol', but not '_atom_site_label'
    if '_atom_site_label' not in pymatgen_dict.keys():
        warnings.warn('W: _atom_site_label not found in parsed dict')
        atom_site_label = []
        symbols = pymatgen_dict['_atom_site_type_symbol']
        for i in range(len(symbols)):
            s = symbols[i]
            atom_site_label.append('{}{}'.format(s, i))
        pymatgen_dict['_atom_site_label'] = atom_site_label
    return identifiers[0], pymatgen_dict


def get_symmop_from_cif_string(cifstring):
    return get_symmop(cifstring2cifdata(cifstring)[1])
