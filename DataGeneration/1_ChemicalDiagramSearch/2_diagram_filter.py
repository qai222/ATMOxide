from collections import Counter

import pandas as pd
from ccdc.io import EntryReader
from tqdm import tqdm

from AnalysisModule.prepare.diagram import ChemicalDiagram, nx, CDFilter, CdBreaker, CdBreakerError
from AnalysisModule.routines.util import read_gcd, yaml_filedump

CSD_READER = EntryReader('CSD')


def cd_filter_function_0(cd: ChemicalDiagram):
    """inclusion: a M is connected to 3 oxygens"""
    for k, v in cd.get_m_env().items():
        if Counter(v)["O"] >= 3:
            return True
    return False


def cd_filter_function_1(cd: ChemicalDiagram):
    """inclusion: one of the connected components contains only C N H"""
    return {"C", "N", "H"} in cd.get_component_element_sets()


def cd_filter_function_2(cd: ChemicalDiagram):
    """exclusion: any M atom connected to non-oxygen or non-halogen atoms"""
    for k, v in cd.get_m_env().items():
        if not set(v).issubset({"O", "Cl", "F", "Br", "I"}):
            return False
    return True


def cd_filter_function_3(cd: ChemicalDiagram):
    """exclusion: all M aquo complex"""
    if cd.all_aquo_complex():
        return False
    return True


def cd_filter_function_5(cd: ChemicalDiagram):
    """exclusion: all M hydroxide"""
    if cd.all_hydroxide():
        return False
    return True


def cd_filter_function_4(cd: ChemicalDiagram):
    """exclusion: less than 2 components"""
    if nx.number_connected_components(cd.graph) < 2:
        return False
    return True


if __name__ == '__main__':

    SubgraphRules = """
    line graph inclusion: any C-N-H,
    line graph exclusion: any O-C-C-C,
    line graph exclusion: any O-C-N,
    line graph exclusion: any O-C-C-N,
    line graph exclusion: any O-C-C-C-N,
    line graph exclusion: any O-C-C-C-C-N,
    line graph exclusion: any P-C-O,
    line graph exclusion: any P-C-C-O,
    line graph exclusion: any P-C-C-C-O,
    line graph exclusion: any O-S-N,
    line graph exclusion: any O-S-C,
    line graph exclusion: any S-C-N,
    line graph exclusion: any P-C-N,
    line graph exclusion: any P-C-C-N,
    line graph exclusion: any P-C-C-C-N,
    line graph exclusion: any P-C-C-C-C-N,
    line graph exclusion: any O-P-C,
    line graph exclusion: any As-C,
    line graph exclusion: any B-C,
    line graph exclusion: any B-N,
    line graph exclusion: any O-N-C,
    line graph exclusion: any O-N-N,
    line graph exclusion: any Si-C,
    line graph exclusion: any M-S-C,
    line graph exclusion: any N-P-O,
    line graph exclusion: any O-P-N,
    line graph exclusion: any O-C-C-O-C-C-O,
    neighbor graph exclusion: any *C has neighbours O;C;C,
    """

    in_subgs, ex_subgs = CDFilter.parse_SubGraphRules(SubgraphRules)

    cdf = CDFilter(
        in_subgs, ex_subgs,
        [
            cd_filter_function_4,
            cd_filter_function_0,
            cd_filter_function_1,
            cd_filter_function_2,
            cd_filter_function_3,
            cd_filter_function_5,
        ]
    )

    input_identifiers = read_gcd("1_sniff_formula.gcd")
    output_identifiers = []
    log_dict = dict()
    for identifier in tqdm(input_identifiers):
        entry = CSD_READER.entry(identifier)
        chemical_diagram = ChemicalDiagram.from_entry(entry)
        accept, log = cdf.accept(chemical_diagram)
        if accept:
            try:
                output = CdBreaker.breakdown_as_jdict(chemical_diagram)
                output_identifiers.append(identifier)
            except CdBreakerError as e:
                log_dict[identifier] = str(e)
            except Exception as e:
                print(identifier, str(e))  # XOMFIO is excluded by rdkit, which is expected
        else:
            log_dict[identifier] = log
    yaml_filedump(log_dict, "2_diagram_filter.yml")
    df = pd.DataFrame(sorted(output_identifiers), columns=['identifier'])
    df.to_csv("2_diagram_filter.gcd", index=False, header=False)
