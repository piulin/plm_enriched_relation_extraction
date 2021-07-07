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
Argparse class: Defines and manages the command-line argument syntax and values.
"""

import argparse
from argparse import RawTextHelpFormatter

class parser(object):

    def __init__(self):
        """
        Initializes the command-line argument parser.
        """

        self.parser = argparse.ArgumentParser(description='Enriched Attention on PTL for Relation Extraction.', formatter_class=RawTextHelpFormatter)

        self.parser.add_argument('--tacred','-t', type=str, help='TACRED dataset.')
        self.parser.add_argument('--batch-size','-b', type=int, help='Sets the batch size for mini-batching training.', default=16)
        self.parser.add_argument('--learning-rate','-l', type=float, help='Sets the learning rate.', default=0.0001)
        self.parser.add_argument('--epochs','-e', metavar='NO_EPHOCS', type=int, help='Sets the number of epochs for '
                                                                                      'mini-batch training.',
                                                                                        default=4)
        self.parser.add_argument('--print-every','-p', metavar='no_iterations', type=int, help='Print loss every '
                                                                                               '`no_iterations` '
                                                                                               'batches.', default=1)

        self.parser.add_argument('--cuda-device','-c', metavar='gpu_id', type=int, default=0,
                                 help='Selects the cuda device. If -1, then CPU is selected.')

        self.parser.add_argument('--experiment-label','-el', metavar='EXECUTION_LABEL', type=str, help='Name an execution', default='Enriched attention PLM')
        self.parser.add_argument('--plm-path','-pp', type=str, help='Path to the pretained langauge model for RoBERTa', default='roberta-base')


    def parse_args(self):

        # Retrieve the arguments as a dictionary
        args = vars(self.parser.parse_args())

        return args