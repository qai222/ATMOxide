from MLModule.encoding import Encode_ms, Encode_nms, Encode_bus, ohe_encode, Encode_smiles, load_input_tables, Encode_elements
import sys

if __name__ == '__main__':


    sys.stdout = open("encode.log", "w")

    """
    encoding condensed dataset, the result is a pickled dataset with its named defined as:
        dataset_<condensation method>_<amine encoding method>.pkl
    e.g.
        dataset_eab_morgan.pkl
        - condensation is performed on indexing fields: elements, amine, building unit
        - elements field is one-hot encoded
        - building unit field is one-hot encoded
        - amine smiles field is encoded using morgan fp
    """
    AmineTable, BuidTable, ElementTable, IdentifierTable = load_input_tables()

    # none condensed, remove amine
    encoding_rules = dict(
        elements=(Encode_elements, dict()),
        bus=(Encode_bus, dict(BuidTable=BuidTable)),
    )
    ohe_encode(
        encoding_rules=encoding_rules,
        condensed_dataset="../1_Condensation/condensed_None.pkl",
        save_as="dataset_none_ablation.pkl"
    )

    # eb condensed, remove amine
    encoding_rules = dict(
        elements=(Encode_elements, dict()),
        bus=(Encode_bus, dict(BuidTable=BuidTable)),
    )
    ohe_encode(
        encoding_rules=encoding_rules,
        condensed_dataset="../1_Condensation/condensed_elements-bus.pkl",
        save_as="dataset_eb_ablation.pkl"
    )

    # eab condensed, hash amine
    encoding_rules = dict(
        elements=(Encode_elements, dict()),
        smiles=(Encode_smiles, dict(nbits=512, funcname="cluster", AmineTable=AmineTable)),
        bus=(Encode_bus, dict(BuidTable=BuidTable)),
    )
    for smiles_encoder_func in (
            'GetHashedAtomPairFingerprintAsBitVect',
            'LayeredFingerprint',
            'GetMACCSKeysFingerprint',
            'GetMorganFingerprintAsBitVect',
            'PatternFingerprint',
            'RDKFingerprint',
            'GetHashedTopologicalTorsionFingerprintAsBitVect',
            "dummy",
            "cluster",
            None,
            "mdes",
            "aemdesTA"
    ):
        encoding_rules["smiles"][1]["funcname"] = smiles_encoder_func
        ohe_encode(
            encoding_rules=encoding_rules,
            condensed_dataset="../1_Condensation/condensed_elements-smiles-bus.pkl",
            save_as="dataset_eab_{}.pkl".format(smiles_encoder_func)
        )
