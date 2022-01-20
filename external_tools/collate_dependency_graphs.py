"""
-------------------------------------------------------------------------------------
Exploring Linguistically Enriched Transformers for Low-Resource Relation Extraction:
    --Enriched Attention on PLM
    
    by Pedro G. Bascoy  (Bosch Center for Artificial Intelligence (BCAI)),
    
    with the supervision of
    
    Prof. Dr. Sebastian Padó (Institut für Machinelle Sprachverarbeitung (IMS)),
    and Dr. Heike Adel-Vu  (BCAI).
-------------------------------------------------------------------------------------
"""

"""
This functions adds external enhanced dependency graphs into an existing TACRED split. The dependency graphs heads appear
as a new key `enhanced_head` in the dictionary for each sample. Check the option [-h] for consulting the command-line 
syntax and further help. 
"""

import argparse
import json
from typing import Dict, List, Any


def parse_args() -> Dict[str, str]:
    """
    Define the command-line arguments. Check the option [-h] to learn more
    :return:
    """
    parser = argparse.ArgumentParser(description='Collate TACRED samples including external dependency parses.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('tacred-split', type=str, help='TACRED split (train, dev, or test) in json format.')

    parser.add_argument('dependency-parse', type=str, help='Dependency parses of the split in CONLLU format.')

    parser.add_argument('out', type=str, help='Collated samples.')

    # Retrieve the  command-line argument as a dictionary
    args = vars(parser.parse_args())

    return args



def main():
    """
    Main function
    :return:
    """

    # Retrieve command-line args.
    args: Dict[str, str] = parse_args()

    # load in memory the tacred dataset
    print('Reading TACRED dataset...')
    tacred:  List[Dict] = load_tacred(args['tacred-split'])
    print('*Done!')

    # load and parse dep. graphs
    print('Parsing CONLLU file...')
    deps: List[List[List[int]]] = load_enhanced_dependency_graphs(args['dependency-parse'])
    print('*Done!')

    # add enhaced dep graphs into the tacred split.
    print('Collating...')
    collate(tacred, deps)
    print('*Done!')


    dump_json(tacred, args['out'])

    print(f'New file created: "{args["out"]}"!')



def dump_json(json_obj: Any,
              path: str) -> None:
    """
    Writes a dictionary into a json file in disk
    :param json_obj: dictionary to be dumped
    :param path: path to the new file
    :return:
    """

    with open(path, 'w') as fp:
        json.dump(json_obj, fp)

def load_enhanced_dependency_graphs(path_to_file: str) -> List[List[List[int]]]:
    """
    This function parses a CONLLU file, and creates returns the list of enhanced dependency heads for each token, sample, and
    sample.
    :param path_to_file: path to the conllu file
    :return: Lists of sizes [samples, tokens, heads]
    """
    with open(path_to_file, 'r', encoding='utf8') as fp:

        # read file content
        lines: List[str] = fp.readlines()

        split_dep_graphs: List[List[List[int]]] = [] # [sample, token, heads]
        sample_dep_graph: List[List[int]] = [] # [token, heads]

        line: str
        for line in lines:

            # if comment detected, skip line
            if line[0] == '#':
                continue

            # if new line detected, we are done with the current sample
            if line == '\n':
                # append sample dep graph to the split list
                split_dep_graphs.append(sample_dep_graph)
                # new sample list
                sample_dep_graph = []

                continue

            # read conllu line content, and parse it
            tok_id: int
            tok: str
            head: int
            dep_type: str
            enhanced_dep_type: str # e.g. 5:nsubj|7:nsubj:xsubj|18:nsubj:xsubj|21:nsubj:xsubj
            tok_id, tok, _, _, _, _, head, dep_type, enhanced_dep_type, _ = line.split('\t')


            # since enhanced dependencies are graphs, we need to keep a list of heads for each token
            enhanced_heads: List[str] = enhanced_dep_type.split('|')
            eheads_list: List[int] = []
            ehead: str
            # get list of heads for the token
            for ehead in enhanced_heads:
                h: int = int( ehead.split(':')[0] )
                eheads_list.append(h)

            sample_dep_graph.append(eheads_list)

    return split_dep_graphs


def load_tacred(path_to_file: str) -> List[Dict]:
    """
    Loads up the tacred split from the json file
    :param path_to_file: json file
    :return:
    """

    # read json file
    with open(path_to_file, 'r') as fp:

        # parse json
        dump: List[Dict] = json.load(fp)

    return dump

def collate(tacred: List[Dict],
            deps: List[List[List[int]]]) -> None:
    """
    Adds the enhanced dependency heads as an attribute to the tacred dictionary
    :param tacred: tacred split
    :param deps: dependency graphs
    :return:
    """

    assert len(tacred) == len(deps)

    i: int
    # iterate samples
    for i in range(len(tacred)):

        # get sample and dep. graph
        tacred_sample: Dict = tacred[i]
        dependency_graph: List[List[int]] = deps[i]

        assert len(tacred_sample['stanford_head']) == len(dependency_graph)

        # add dep. graph into the tacred sample
        tacred_sample ['enhanced_head'] = dependency_graph




if __name__ == '__main__':
    # main function
    main()
