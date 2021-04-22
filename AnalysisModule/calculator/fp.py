import logging

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import MolFromSmiles
from rdkit.DataStructs import ConvertToNumpyArray


def Rdkit1DFp():
    fp1dnames = []
    fp1dfuncnames = []
    for funcname in sorted(dir(AllChem), key=lambda x: x.endswith("AsBitVect"), reverse=True):
        fpname = funcname.replace("AsBitVect", "").replace("Get", "").replace("Hashed", "").replace("Fingerprint", "")
        if fpname not in fp1dnames:
            if funcname.endswith("AsBitVect") and "Fingerprint" in funcname and callable(getattr(AllChem, funcname)):
                fp1dnames.append(fpname)
                fp1dfuncnames.append(funcname)
            elif funcname.endswith("Fingerprint") and fpname not in fp1dnames:
                if "erg" in funcname.lower():  # skip 2d
                    continue
                fp1dnames.append(fpname)
                fp1dfuncnames.append(funcname)
    return dict(zip(fp1dnames, fp1dfuncnames))


def GetFp(funcname, smiles: str, nbits=512):
    mol = MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    # rdkit1d = Rdkit1DFp()
    func = getattr(AllChem, funcname)
    if funcname.endswith("AsBitVect"):
        try:
            fp = func(mol, nBits=nbits)
        except:
            fp = func(mol, nBits=nbits, radius=2)  # morgan needs radius
    else:
        try:
            fp = func(mol, fpSize=nbits)
        except:
            fp = func(mol)  # MACC is fixed
    res = np.array(0)
    ConvertToNumpyArray(fp, res)
    res = res.astype(np.uint8)
    return res


def GetFpArray(smiles: [str], nbits: int, funcname: str):
    if "maccs" in funcname.lower():
        nbits = 167
    fparray = np.zeros((len(smiles), nbits), dtype=np.uint8)
    ismi = 0
    for smi in smiles:
        fparray[ismi] = GetFp(funcname, smi, nbits)
        ismi += 1

    logging.info("GetFpArray output nbits: {}".format(nbits))
    # for ismi in range(len(smiles)):
    #     fparray[ismi] = GetFp(funcname, smiles[ismi], nbits)
    return fparray
