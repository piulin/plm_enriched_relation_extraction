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
# parse command-line arguments
def parse_arguments():

    parser = argparse.ArgumentParser(description='Dataset resize utility.')
    parser.add_argument('dataset_json_file', type=str, help='Path to the dataset.')

    return vars(parser.parse_args())

if __name__ == '__main__':

    # parse command-line args.
    args: Dict[str, Any] = parse_arguments()
    relations: Dict[str,int] = {}
    # read content of dataset
    with open(args['dataset_json_file'],'r') as fp:
        dump: List[Dict]= json.load(fp)


        print(f'Size of the dataset: {len(dump)}')

        for sample in dump:


            if sample['relation'] in relations:
                relations[sample['relation']] += 1
            else:
                relations[sample['relation']] = 0


        for rel_name, rel_count in relations.items():

            print(f'Relation: {rel_name}, count: {rel_count}')
