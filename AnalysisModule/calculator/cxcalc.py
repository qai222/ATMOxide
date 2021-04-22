import subprocess
from io import StringIO

import pandas as pd

from AnalysisModule.routines.util import removefile

"""
a wrapper for jchemsuite
"""

Jia2019Si = """_feat_Mass
Molecular mass of _raw_SMILES (possibly a salt)

_feat_AtomCount_C
(atomcount -z 6)
Number of carbon atoms. (integer)

_feat_AtomCount_N
(atomcount -z 7)
Number of nitrogen atoms. (integer)

_feat_AvgPol 
Average molecular polarizability (at _rxn_pH)

_feat_MolPol 
Molecular polarizability (at _rxn_pH)

_feat_Refractivity 
Computed refractivity

_feat_isoelectric
(isoelectricpoint)
Isoelectric point of the molecule

_feat_AliphaticRingCount
Number of aliphatic rings (integer)

_feat_AromaticRingCount 
Number of aromatic rings (integer)

_feat_AliphaticAtomCount
AtomCount Number of aliphatic atoms in the molecule (integer)

_feat_AromaticAtomCount
Number of aromatic atoms in the molecule (integer)

_feat_BondCount 
Number of bonds in the molecule (integer)

_feat_CarboaliphaticRingCount 
Number of aliphatic rings comprised solely of carbon atoms (integer)

_feat_CarboaromaticRingCount 
Number of aromatic rings comprised solely of carbon atoms (integer)

_feat_CarboRingCount 
Number of rings comprised solely of carbon atoms (integer)

_feat_ChainAtomCount 
Number atoms that are part of chain (not part of a ring) (integer)

_feat_ChiralCenterCount 
Number of tetrahedral stereogenic centers (integer)

_feat_RingAtomCount 
Number of atoms that are part of a ring (not part of a chain) (integer)

_feat_SmallestRingSize 
Number of members in the smallest ring (integer)

_feat_LargestRingSize 
Number of members in the largest ring (integer)

_feat_fsp3 
Fraction of sp3 carbons (Fsp3 value)

_feat_HeteroaliphaticRingCount 
Number of heteroaliphatic rings (integer)

_feat_HeteroaromaticRingCount
Number of heteroaromatic rings (integer)

_feat_RotatableBondCount
Number of rotatable bonds (integer)

_feat_BalabanIndex
Balaban molecular graph index

_feat_CyclomaticNumber 
Cyclomatic number of molecular graph

_feat_HyperWienerIndex 
Hyper Wiener Index of molecular graph

_feat_WienerIndex 
Wiener Index of molecular graph

_feat_WienerPolarity 
Wiener Polarity of molecular graph

_feat_MinimalProjectionArea 
Minimal projection area

_feat_MinimalProjectionRadius 
Minimal projection radius

_feat_MinimalProjectionRadius 
Minimal projection radius

_feat_MaximalProjectionRadius 
Maximal projection radius

_feat_LengthPerpendicularToTheMinArea
(minimalprojectionsize)
Length perpendicular to the minimal projection area

_feat_LengthPerpendicularToTheMaxArea
(maximalprojectionsize)
Length perpendicular to the maximum projection area

_feat_VanderWaalsVolume
(volume)
van der Waals volume of the molecule

_feat_VanderWaalsSurfaceArea
(vdwsa)
van der Waals surface area of the molecule

_feat_ASA
(asa -H _rxn_pH)
Water accessible surface area of the molecule, computed at _rxn_pH

_feat_ASA+
(molecularsurfacearea -t ASA+ -H _rxn_pH)
Water accessible surface area of all atoms with positive partial charge,computed at _rxn_pH

_feat_ASA-
(molecularsurfacearea -t ASA- -H _rxn_pH)
Water accessible surface area of all atoms with negative partial charge,computed at _rxn_pH

_feat_ASA_H
(molecularsurfacearea -t ASA_H -H _rxn_pH)
Water accessible surface area of all hydrophobic atoms with positive partial charge,computed at _rxn_pH

_feat_ASA_P
(molecularsurfacearea -t ASA+P -H _rxn_pH)
Water accessible surface area of all polar atoms with positive partial charge,computed at _rxn_pH

_feat_PolarSurfaceArea
(polarsurfacearea -H _rxn_pH)
2D Topological polar surface area, computed at _rxn_pH

_feat_acceptorcount
(acceptorcount -H _rxn_pH)
Hydrogen bond acceptor atom count in molecule, computed at _rxn_pH

_feat_Accsitecount
(acceptorsitecount -H _rxn_pH)
Hydrogen bond acceptor multiplicity in molecule, computed at _rxn_pH

_feat_donorcount
(donorcount -H _rxn_pH)
Hydrogen bond donor atom count in molecule, computed at _rxn_pH

_feat_donsitecount 
Hydrogen bond donor multiplicity in molecule, computed at _rxn_pH

_feat_sol 
(solubility -H _rxn_pH)
Aqueous solubility (logS) computed at _rxn_pH

_feat_apKa
(pka -a 2)
First and second acidic pKa value. Subsequent columns are the subsequent entries in the returned list.

_feat_bpKa1
(pka -b 4)
First-fourth basic pKa value. Subsequent columns are the"""


class CxFeature:
    def __init__(self, feature: str, dstring: str, comment: str):
        self.feature = feature
        self.dstring = dstring
        self.comment = comment

    def __repr__(self):
        return self.dstring


from collections import OrderedDict


def get_Feautre2Dstring(si_string: str):
    des_strings = si_string.split("\n\n")
    fd = OrderedDict()
    for des_string in des_strings:
        lines = des_string.split("\n")
        if len(lines) == 2:
            feature = lines[0][6:].strip().lower()
            comment = lines[1]
            if "rxn_ph" in comment.lower():
                dstring = "{} -H _rxn_pH".format(feature)
            else:
                dstring = feature
        elif len(lines) == 3:
            feature = lines[0].strip("_feat_").strip().lower()
            dstring = lines[1].strip().strip("(").strip(")")
            comment = lines[2]
            if "rxn_ph" in comment.lower() and "rxn_ph" not in dstring.lower():
                dstring = "{} -H _rxn_pH".format(dstring)
        else:
            raise ValueError("{}\n cannot be parsed!".format(des_string))
        fd[feature] = CxFeature(feature, dstring, comment)
    return fd


Jia2019FeatureDict = get_Feautre2Dstring(Jia2019Si)


class JChemCalculator:

    def __init__(self, cxexe="/home/ai/localpkg/jchemsuite/bin/cxcalc"):
        self.cxexe = cxexe

    def cal_feature(self, features: [CxFeature], smiles: [str], rmtmp=True, rxnph: float = None):
        instring = "\n".join(smiles)
        tmpfn = "jchem" + str(hash(instring)) + ".smiles"
        with open(tmpfn, "w") as f:
            f.write(instring)

        if rxnph is None:
            dstring = " ".join([cf.dstring for cf in features if "rxn_ph" not in cf.dstring.lower()])
        elif isinstance(rxnph, float):
            dstring = " ".join([cf.dstring for cf in features])
            dstring.replace("_rxn_pH", str(rxnph))
        else:
            raise ValueError("rxnph must be a float or None!: {}".format(rxnph))

        # dnames
        cmd = "{} {} {}".format(self.cxexe, dstring, tmpfn)
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
        (out, err) = proc.communicate()
        out = out.decode("utf-8")
        df = pd.read_csv(StringIO(out), sep="\t")
        if rmtmp:
            removefile(tmpfn)
        return df
