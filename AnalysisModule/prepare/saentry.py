from pymatgen.core.structure import Structure


class SaotoEntry:

    def __init__(self, identifier: str, details: dict, clean_structure: Structure, disordered_structure: Structure):
        self.identifier = identifier
        self.details = details
        self.clean_structure = clean_structure
        self.disodered_structure = disordered_structure

    def as_dict(self):
        d = dict(
            identifier=self.identifier,
            details=self.details,
            clean_structure=self.clean_structure.as_dict(),
            disordered_structure=self.disodered_structure.as_dict()
        )
        return d

    @classmethod
    def from_dict(cls, d):
        return cls(d["identifier"], d["details"], d["clean_structure"], d["disordered_structure"])
