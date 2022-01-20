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

import argparse
import json
from typing import List, Any, Dict, Tuple, Union, Optional
import random
from datetime import datetime

"""ace2tacred.py utility. Converts from JSON ACE05 format (see https://github.com/luanyi/DyGIE/tree/master/preprocessing) into TACRED format.
"""

# parse command-line arguments
def parse_arguments():

    parser = argparse.ArgumentParser(description='Dataset resize utility.')
    parser.add_argument('dataset_json_file', type=str, help='Path to the ACE05 json dataset.')
    parser.add_argument('output', type=str, help='Output dataset in tacred format.')

    return vars(parser.parse_args())

if __name__ == '__main__':

    # parse command-line args.
    args: Dict[str, Any] = parse_arguments()

    # JSON output
    out: List[Dict] = []
    no_relations = 0

    # read content of dataset
    with open(args['dataset_json_file'],'r') as fp:

        documents = fp.readlines()

        for document in documents:

            token_idx : int = 0
            dump: List[Dict]= json.loads(document)

            sentences: List[List[str]] = dump['sentences']
            relations: List[List[List]] = dump['relations']
            ners: List[List[List]] = dump['ner']
            dep_heads: Dict[str,int] = dump['dep_head']


            for sentence_idx, sentence in enumerate(sentences):


                # If there is no relation, we won't include it in the output.
                if not relations[sentence_idx]:
                    pass
                else:

                    #loop relations
                    for relation in relations[sentence_idx]:
                        no_relations += 1
                        relation_dict: Dict = {}
                        relation_dict['token']: List[str] = sentence
                        relation_dict['relation']: str = relation[4]
                        relation_dict['subj_start']: int = relation[0] - token_idx
                        relation_dict['subj_end']: int = relation[1] - token_idx
                        relation_dict['obj_start']: int = relation[2] - token_idx
                        relation_dict['obj_end']: int = relation[3] - token_idx
                        stanford_head: List[int] = []

                        for i in range(len(sentence)):

                            global_token_id: int = i + token_idx
                            global_head_id: int = dep_heads[str(global_token_id)]
                            head_id: int = 0

                            # if head is root
                            if (global_head_id != 0):
                                head_id = global_head_id - token_idx + 1

                            stanford_head.append(head_id)
                        relation_dict['stanford_head'] = stanford_head


                        stanford_ner: List[str] = ['O' for i in range(len(sentence))]
                        for ner in ners[sentence_idx]:
                            for i in range(ner[0],ner[1]+1):
                                token_id = i - token_idx
                                stanford_ner[token_id] = ner[2]
                        relation_dict['stanford_ner'] = stanford_ner

                        out.append(relation_dict)

                token_idx += len(sentence)


        print(f'No relations exported: {no_relations}')
            # export to file
        with open(args['output'],'w') as fout:
            json.dump(out, fout, indent=4)



