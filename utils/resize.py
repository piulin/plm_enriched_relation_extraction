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

"""Resize.py utility. It allows to take a dataset and resized to a percetange of samples of the total amount.
"""

# parse command-line arguments
def parse_arguments():

    parser = argparse.ArgumentParser(description='Dataset resize utility.')

    parser.add_argument('dataset_json_file', type=str, help='Path to the dataset to be resized.')
    parser.add_argument('percentage', type=float, help='Percentage of the dataset to be exported. It should range between 0.0 and 1.0')
    parser.add_argument('output', type=str, help='Output dataset.')

    return vars(parser.parse_args())

if __name__ == '__main__':

    # parse command-line args.
    args: Dict[str, Any] = parse_arguments()

    # read content of dataset
    with open(args['dataset_json_file'],'r') as fp:
        dump: List[Dict]= json.load(fp)

        # determine #samples
        length : int = len(dump)
        percentage: float = args['percentage']

        new_sample_size: float = int( length*percentage )
        print(f'Input dataset size: {length}, Output dataset size: {new_sample_size}')

        # set random seed
        random.seed(56843321)

        # shuffle first level of json dict
        random.shuffle(dump)

        # Select first `new_sample_size` samples from the shuffled dataset.
        out: List[Dict] = dump[0:new_sample_size]

        # export to file
        with open(args['output'],'w') as fout:
            json.dump(out, fout)



