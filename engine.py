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
from datasets.dataset import dataset
from datasets.tacred.tacred import tacred
from datasets.tacred.tacred_emt import tacred_emt
from datasets.tacred.tacred_enriched import tacred_enriched
from ai.re import re
from utils import utils
from tokenizer import tokenizer
import torch
import random
from typing import Dict, Union, List, Tuple
from log.teletype import teletype


def run(args : dict) -> None:
    """
    This is where the program starts after command-line argument parsing.
    It handles the main logic of the classifier
    :param args: command-line arguments
    :return:
    """

    teletype.start_task('Configuring engine',__name__)

    # Config seeds
    setup_seeds(args['seed'])

    # Retrieves the device that will perform the training/classification (either CPU or GPU)
    device: torch.device = utils.get_device(args['cuda_device'])

    # Create output folders if they do not exist
    utils.create_folder(args['figure_folder'])

    # Initialize a tokenizer and ID mapper (for converting tokens to IDs in the datasets)
    tokzer: tokenizer = tokenizer.tokenizer( args['plm_model_path'] )

    # get splits (https://www.python.org/dev/peps/pep-0526/#global-and-local-variable-annotations)
    train: dataset
    dev: dataset
    test: dataset
    train, dev, test = get_datasets(args, tokzer, device)

    # add remainder args to `args`
    add_arguments(args, train, device, tokzer)

    # Initializes the neural network
    model: re = re (args=args, **args)

    teletype.finish_task(__name__)

    # Train
    model.fit(train,
              args['batch_size'],
              args['learning_rate'],
              args['print_every'],
              args['epochs'],
              args['no_eval_batches'],
              dev_dataset=dev
              )



    # Evaluate on test
    model.evaluate(test,
                   args['batch_size'],
                   'Test',
                   no_batches=None,
                   plot=True
                   )





def setup_seeds(seed: Union[int, None]) -> None:
    """
    Sets seed for deterministic operation
    :param seed:
    :return:
    """

    teletype.start_subtask(f'Configuring seed: {seed}', __name__, 'setup_seeds')

    if seed is not None:

        # Configure seeds
        torch.manual_seed(seed)
        random.seed(seed)
        numpy.random.seed(seed)
        torch.use_deterministic_algorithms(True)

        teletype.finish_subtask( __name__, 'setup_seeds')
    else:
        teletype.finish_subtask( __name__, 'setup_seeds', message='No seed provided')


def get_datasets(args:  dict,
                 tokzer : tokenizer,
                 device: torch.device) -> Tuple[dataset, dataset, dataset] :
    """
    Retrieves the datasets from disk
    :param args: command-line arguments
    :param tokzer: tokenizer
    :param device: torch device id
    :return: train, dev, and test splits
    """

    # If tacred option is provided, load it
    if args['tacred']:

        # Retrieve the class of the dataset to be used given the schema
        schema_switcher: dict = {
            'ess': tacred_emt,
            'standard': tacred,
            'enriched_attention': tacred_enriched
        }
        dataset_class = schema_switcher.get(args['schema'], None)

        # retrieve the paths to the train, test, and development json
        train_json_path: str
        dev_json_path: str
        test_json_path: str
        train_json_path, dev_json_path , test_json_path = tacred.build_file_paths(args['tacred'], args['mini_dataset'])

        # load data splits
        train: dataset_class = dataset_class ( train_json_path, tokzer , device )

        # for dev a test dataset, we also pass the relation mapper so relations classes share the same IDs.
        dev: dataset_class = dataset_class ( dev_json_path, tokzer, device, train.relation_mapper, train.ner_mapper )
        test: dataset_class = dataset_class ( test_json_path, tokzer, device, train.relation_mapper, train.ner_mapper )

        return train, dev, test


def add_arguments(args: dict,
                  train: dataset,
                  device: torch.device,
                  tokzer: tokenizer) -> None:
    """
    Adds remaining arguments
    :param args: argument dictionary
    :param train: train dataset
    :param device: network computing device
    :param tokzer: tokenizer
    :return:
    """

    args['number_of_relations'] = train.get_number_of_relations()
    args['device'] = device
    args['vocabulary_length'] = len(tokzer)
    # +1 accounts for the padding idx, and another +1 because the 0 value also counts as a slot
    # note, therefore, that the padding_idx would be `train.highest_token_entity_distance() + 1`
    args['num_position_embeddings'] = train.highest_token_entity_distance() + 1 + 1
    # +1 accounts for the padding idx, and another +1 because the 0 value also counts as a slot
    args['num_dependency_distance_embeddings'] = train.highest_dependency_entity_distance() + 1 + 1

    args['num_entity_embeddings'] = train.get_number_of_entity_types()

