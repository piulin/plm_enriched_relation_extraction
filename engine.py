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
Engine module: it is responsible of the main logic of the program.
"""

# from datasets import dataset
from datasets.tacred.tacred import tacred
from ai.re import re
from utils import utils
from tokenizer import tokenizer

def run(args):
    """
    This is where the program starts after command-line argument parsing.
    It handles the main logic of the classifier
    :param args: command-line arguments
    :return:
    """

    # Retrieves the device that will perform the training/classification (either CPU or GPU)
    device = utils.get_device(args['cuda_device'])

    # Initialize a tokenizer and ID mapper (for converting tokens to IDs in the datasets)
    tokzer = tokenizer.tokenizer( args['plm_path'] )

    # Train dataset
    train = None

    # If tacred option is provided, load it
    if args['tacred']:

        # retrieve the paths to the train, test, and development json
        train_json_path, _, _ = tacred.build_file_paths(args['tacred'])

        # load the train split
        train = tacred ( train_json_path, tokzer , device )


    # Initializes the neural network
    model = re ( train.get_number_of_relations(),
                 device,
                 args['plm_path'],
                 args )

    # Train
    model.fit(train,
              args['batch_size'],
              args['learning_rate'],
              args['print_every'],
              args['epochs']
              )


