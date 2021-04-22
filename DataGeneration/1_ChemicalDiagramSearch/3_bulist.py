import pandas as pd
from ccdc.io import EntryReader
from tqdm import tqdm

from AnalysisModule.prepare.diagram import BuildingUnit
from AnalysisModule.prepare.diagram import ChemicalDiagram, CdBreaker
from AnalysisModule.routines.util import read_gcd, graph_json_loads, save_jsonfile, MontyEncoder

CSD_READER = EntryReader('CSD')

if __name__ == '__main__':
    hitlist = sorted(read_gcd("2_diagram_filter.gcd"))
    UniqueBus = set()
    BUid = 0
    records = []

    for identifier in tqdm(hitlist):
        entry = CSD_READER.entry(identifier)
        chemical_diagram = ChemicalDiagram.from_entry(entry)
        output = CdBreaker.breakdown_as_jdict(chemical_diagram)
        record = {}
        record["identifier"] = identifier
        smis = output["smis"]
        if len(set(smis)) > 1:
            print("more than one amine?:", identifier, smis)
        record["smiles"] = smis[0]

        momgs = output["momgs"]
        momgs = [graph_json_loads(mgj) for mgj in momgs]
        bus_in_this_crystal = set()
        for mg in momgs:
            bus = BuildingUnit.GetBuildingUnits(mg)
            for bu in bus:
                if not bu.is_allowed:
                    continue

                this_bu_found = False
                for bu_found in UniqueBus:
                    if bu_found == bu:
                        this_bu_found = True
                        break
                if this_bu_found:
                    bu.buid = bu_found.buid
                    bu.cdg = chemical_diagram.graph
                    bus_in_this_crystal.add(bu)
                else:
                    bu.buid = BUid
                    bu.cdg = chemical_diagram.graph
                    UniqueBus.add(bu)
                    BUid += 1
                    bus_in_this_crystal.add(bu)
        record["bus"] = [bu.buid for bu in bus_in_this_crystal]
        records.append(record)
    UniqueBus = sorted(UniqueBus, key=lambda x: x.buid)

    save_jsonfile(UniqueBus, "3_bulist.json", encoder=MontyEncoder)

    outf = open("3_bulist.html", "w")
    for bu in UniqueBus:
        bu: BuildingUnit
        svgtext = bu.draw_by_cdg(title=bu.buid)
        outf.write(svgtext)
    outf.close()

    df = pd.DataFrame.from_records(records)
    df.to_csv("3_bulist.csv", index=False)
