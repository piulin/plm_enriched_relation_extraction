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
import numpy.random

"""
Engine module: it is responsible of the main logic of the program.
"""

# from datasets import dataset
from datasets.tacred.tacred import tacred
from datasets.tacred.tacred_emt import tacred_emt
from datasets.tacred.tacred_enriched import tacred_enriched
from ai.re import re
from utils import utils
from tokenizer import tokenizer
import torch
import random

def run(args):
    """
    This is where the program starts after command-line argument parsing.
    It handles the main logic of the classifier
    :param args: command-line arguments
    :return:
    """

    # Config seeds
    setup_seeds(args['seed'])

    # Retrieves the device that will perform the training/classification (either CPU or GPU)
    device = utils.get_device(args['cuda_device'])

    # Create output folders if they do not exist
    utils.create_folder(args['fig_folder'])

    # Initialize a tokenizer and ID mapper (for converting tokens to IDs in the datasets)
    tokzer = tokenizer.tokenizer( args['plm_path'] )

    # get splits
    train, dev, test = get_datasets(args, tokzer, device)

    # Initializes the neural network
    model = re ( train.get_number_of_relations(),
                 device,
                 args['plm_path'],
                 args['fig_folder'],
                 len(tokzer),
                 args['schema'],
                 args,
                 # +1 accounts for the padding idx, and another +1 because the 0 value also counts as a slot
                 # note, therefore, that the padding_idx would be `train.highest_token_entity_distance() + 1`
                 train.highest_token_entity_distance() + 1 + 1,
                 # TODO: move to kwargs...
                 args['position_distance_embedding_size'] if 'position_distance_embedding_size' in args else None,
                 # +1 accounts for the padding idx, and another +1 because the 0 value also counts as a slot
                 train.highest_dependency_entity_distance() + 1 + 1,
                 args['position_distance_embedding_size'] if 'position_distance_embedding_size' in args else None,
                 args['attention_size'] if 'attention_size'in args else None)


    # Train
    model.fit(train,
              args['batch_size'],
              args['learning_rate'],
              args['print_every'],
              args['epochs'],
              dev_dataset=dev
              )



    # Evaluate on test
    model.evaluate(test,
                   args['batch_size'],
                   'Test',
                   no_batches=None,
                   plot=True
                   )





def setup_seeds(seed):
    """
    Sets sees for deterministic operation
    :param seed:
    :return:
    """
    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed)
        numpy.random.seed(seed)
        torch.use_deterministic_algorithms(True)


def get_datasets(args,
                 tokzer,
                 device):
    """
    Retrieves the datasets from disk
    :param args: command-line arguments
    :param tokzer: tokenizer
    :param device: torch device id
    :return: train, dev, and test splits
    """
    # Datasets
    train = None
    dev = None
    test = None



    # If tacred option is provided, load it
    if args['tacred']:

        # Retrieve the class of the dataset to be used given the schema
        schema_switcher = {
            'ESS': tacred_emt,
            'Standard': tacred,
            'Enriched_Attention': tacred_enriched
        }
        dataset_class = schema_switcher.get(args['schema'], None)

        # retrieve the paths to the train, test, and development json
        train_json_path, dev_json_path , test_json_path = tacred.build_file_paths(args['tacred'])

        # load data splits
        train = dataset_class ( train_json_path, tokzer , device )

        # for dev a test dataset, we also pass the relation mapper so relations classes share the same IDs.
        dev = dataset_class ( dev_json_path, tokzer, device, train.relation_mapper )
        test = dataset_class ( test_json_path, tokzer, device, train.relation_mapper )

    return train, dev, test