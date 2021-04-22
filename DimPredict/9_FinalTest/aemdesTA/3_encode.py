from MLModule.encoding import ohe_encode, load_input_tables, Encode_nms, Encode_ms, Encode_smiles, Encode_bus, Encode_elements


AmineTable, BuidTable, ElementTable, IdentifierTable = load_input_tables()

encoding_rules = dict(
    elements=(Encode_elements, dict()),
    smiles=(Encode_smiles, dict(nbits=512, funcname="aemdesTAextra", AmineTable=AmineTable)),
    bus=(Encode_bus, dict(BuidTable=BuidTable)),
)
ohe_encode(
    encoding_rules=encoding_rules,
    condensed_dataset="./condensed.pkl",
    save_as="./dataset_eab_aemdes.pkl"
)
