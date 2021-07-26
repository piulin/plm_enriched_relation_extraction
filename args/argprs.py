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
from typing import Dict, Union

"""
Argparse class: Defines and manages the command-line argument syntax and values. Check [-h] to learn more.
"""

import argparse
from argparse import ArgumentDefaultsHelpFormatter
from datetime import datetime

class parser(object):

    def __init__(self):
        """
        Initializes the command-line argument parser.
        """

        self.parser = argparse.ArgumentParser(description='Enriched Attention on PLM for Relation Extraction.',
                                              formatter_class=ArgumentDefaultsHelpFormatter)

        self.parser.add_argument('--batch-size','-b', type=int, help='Sets the batch size for mini-batching training.', default=16)

        self.parser.add_argument('--cuda-device','-c', metavar='gpu_id', type=int, default=0,
                                 help='Selects the cuda device. If -1, then CPU is selected.')


        self.subparsers = self.parser.add_subparsers(title='Mode',help='Action to perform')

        self.configure_train_parser()


    def configure_train_parser(self) -> None:
        """
        Configures the train subparser
        :return:
        """


        self.train_parser = self.subparsers.add_parser('train', help='Train the predictor', formatter_class=ArgumentDefaultsHelpFormatter)

        self.train_parser.set_defaults(action='train')

        self.train_parser.add_argument('--experiment-label', '-el', metavar='EXECUTION_LABEL', type=str,
                                       help='Name the execution.',
                                       default='Enriched attention PLM')

        self.train_parser.add_argument('--run-label', '-rl', metavar='RUN_LABEL', type=str, help='Name the run.',
                                       default=datetime.today().strftime('%Y-%m-%d'))

        self.train_parser.add_argument('--disable-mlflow', '-dm',
                                       action='store_true',
                                       help='If used, the program will not log performance metrics into mlflow.',
                                       default=False)

        self.train_parser.add_argument('--tacred', '-t', type=str, help='TACRED dataset.', required=True)
        self.train_parser.add_argument('--learning-rate', '-l', metavar=('PLM', 'PTL'), type=float,
                                       help='Sets the learning rates.', nargs=2, default=[1e-3, 5e-5])

        self.train_parser.add_argument('--epochs', '-e', metavar='NO_EPHOCS', type=int,
                                       help='Sets the number of epochs for '
                                            'mini-batch grad desc.',
                                       default=7)
        self.train_parser.add_argument('--print-every', '-p', metavar='no_iterations', type=int,
                                       help='Print loss every '
                                            '`no_iterations` '
                                            'batches.', default=1)

        self.train_parser.add_argument('--plm-model-path', '-pmp', type=str,
                                       help='Path to the pretrained language  model for RoBERTa.', default='roberta-base')

        self.train_parser.add_argument('--figure-folder', '-ff', type=str,
                                       help='Path to the folder where figures will be saved.', default='figures/')

        self.train_parser.add_argument('--seed', '-s', type=int, help='Set a seed for pytorch.', default=None)

        subparser = self.train_parser.add_subparsers(title='schema', help='Select the training schema.')

        self.standard_parser = subparser.add_parser('Standard', help='Use Standard+CLS token as training schema',
                                                    formatter_class=ArgumentDefaultsHelpFormatter)
        self.standard_parser.set_defaults(schema='Standard')
        self.ess_parser = subparser.add_parser('ESS', help='Use EMT+ESS token as training schema',
                                               formatter_class=ArgumentDefaultsHelpFormatter)
        self.ess_parser.set_defaults(schema='ESS')

        self.enriched_attention_parser = subparser.add_parser('Enriched_Attention', help='Use enriched attention.',
                                                              formatter_class=ArgumentDefaultsHelpFormatter)
        self.enriched_attention_parser.set_defaults(schema='Enriched_Attention')

        self.enriched_attention_parser.add_argument('--dependency-distance-size', '-des', type=int,
                                                    help='Size of the dependency distance embeddings.',
                                                    default=16)

        self.enriched_attention_parser.add_argument('--position-embedding-size', '-pes', type=int,
                                                    help='Size of the token distance embeddings.',
                                                    default=16)

        self.enriched_attention_parser.add_argument('--attention-size', '-as', type=int,
                                                    help='Number of neurons of the attention layer.',
                                                    default=128)

    def parse_args(self) -> Dict[str, Union[int, str, float]] :

        # Retrieve the arguments as a dictionary
        args = vars(self.parser.parse_args())

        return args