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


        self.subparsers = self.parser.add_subparsers(title='Mode',help='Action to perform',dest='action', required=True)

        self.configure_train_parser()


    def configure_train_parser(self) -> None:
        """
        Configures the train subparser
        :return:
        """


        self.train_parser = self.subparsers.add_parser('train', help='Train the predictor', formatter_class=ArgumentDefaultsHelpFormatter)

        self.train_parser.set_defaults(action='train')

        self.train_parser.add_argument('--dropout-probability', '-dp', type=float,
                                       help='Dropout probability for the regularization layers.',
                                       default=0.5)

        self.train_parser.add_argument('--experiment-label', '-el', metavar='EXECUTION_LABEL', type=str,
                                       help='Name the execution.',
                                       default='Enriched attention PLM')

        self.train_parser.add_argument('--optimizer', '-op', metavar='OPTIMIZER_NAME', type=str,
                                       choices=['Adam',
                                                'AdamW',
                                                'SDG',
                                                'Adamax',
                                                'MyAdagrad'],
                                       help='Select an optimizer: "Adam", "AdamW", "SDG", "Adamax" or "MyAdagrad"',
                                       default='AdamW')

        self.train_parser.add_argument('--init-method', '-im', metavar='INIT_NAME', type=str,
                                       choices=["none",
                                                "xavier_uniform",
                                                "xavier_normal",
                                                "kaiming_uniform_fan_in",
                                                "kaiming_uniform_fan_out",
                                                "kaiming_normal_fan_in",
                                                "kaiming_normal_fan_out"],
                                       help='Select an initializaton method: "none", "xavier_uniform", '
    '"xavier_normal", "kaiming_uniform_fan_in", "kaiming_uniform_fan_out", "kaiming_normal_fan_in" or "kaiming_normal_fan_out"',
                                       default='none')

        self.train_parser.add_argument('--scheduler', '-sc', metavar='SCHEDULER_NAME', type=str,
                                       choices=['linear',
                                                'cosine',
                                                'cosine_with_restarts',
                                                'polynomial',
                                                'constant',
                                                'constant_with_warmup'],
                                       help='Select a scheduler: "linear", "cosine", "cosine_with_restarts", "polynomial", "constant" or "constant_with_warmup"',
                                       default='linear')

        self.train_parser.add_argument('--run-label', '-rl', metavar='RUN_LABEL', type=str, help='Name the run.',
                                       default=datetime.today().strftime('%Y-%m-%d'))

        self.train_parser.add_argument('--disable-mlflow', '-dm',
                                       action='store_true',
                                       help='If used, the program will not log performance metrics into mlflow.',
                                       default=False)

        self.train_parser.add_argument('--tacred', '-t', type=str, help='TACRED dataset.', required=True)

        self.train_parser.add_argument('--mini-dataset', '-md',
                                       action='store_true',
                                       help='If used, train, test, and development datasets will point to a toy '
                                            'corpus (for development purposes).',
                                       default=False)

        self.train_parser.add_argument('--percentage',
                                       type=float,
                                       help='If used, train, test, and development datasets will point to a file with `percentage` of the training data. The file must exist in the dataset directory.',
                                       default=1.0)

        self.train_parser.add_argument('--enhanced-dependencies', '-ed',
                                       action='store_true',
                                       help='If used, train, test, and development datasets will be loaded with its enhanced dependencies version.',
                                       default=False)

        self.train_parser.add_argument('--no-eval-batches','-neb', type=int,
                                       help='Configures the number of random batches assessed every `print_every` iterations',
                                       default=30)

        self.train_parser.add_argument('--learning-rate', '-l', metavar=('PLM', 'PTL'), type=float,
                                       help='Sets the learning rates.', nargs=2, default=[1e-3, 5e-5])

        self.train_parser.add_argument('--epochs', '-e', metavar='NO_EPHOCS', type=int,
                                       help='Sets the number of epochs for '
                                            'mini-batch grad desc.',
                                       default=7)
        self.train_parser.add_argument('--print-every', '-p', metavar='no_iterations', type=int,
                                       help='Print loss every '
                                            '`no_iterations` '
                                            'batches.', default=400)

        self.train_parser.add_argument('--plm-model-path', '-pmp', type=str,
                                       help='Path to the pretrained language  model for RoBERTa.', default='roberta-base')

        self.train_parser.add_argument('--figure-folder', '-ff', type=str,
                                       help='Path to the folder where figures will be saved.', default='figures/')

        self.train_parser.add_argument('--seed', '-s', type=int, help='Set a seed for pytorch.', default=None)

        subparser = self.train_parser.add_subparsers(title='schema', help='Select the training schema.', dest='schema', required=True)

        self.standard_parser = subparser.add_parser('standard', help='Use Standard+CLS token as training schema',
                                                    formatter_class=ArgumentDefaultsHelpFormatter)
        self.standard_parser.set_defaults(schema='standard')
        self.ess_parser = subparser.add_parser('ess', help='Use EMT+ESS token as training schema',
                                               formatter_class=ArgumentDefaultsHelpFormatter)
        self.ess_parser.set_defaults(schema='ess')

        self.enriched_attention_parser = subparser.add_parser('enriched_attention', help='Use enriched attention.',
                                                              formatter_class=ArgumentDefaultsHelpFormatter)
        self.enriched_attention_parser.set_defaults(schema='enriched_attention')

        self.enriched_attention_parser.add_argument('--attention-size', '-as', type=int,
                                                    help='Number of neurons of the attention layer.',
                                                    default=128)

        self.enriched_attention_parser.add_argument('--dependency-distance-size', '-des', type=int,
                                                    help='Size of the dependency distance embeddings.',
                                                    default=16)

        attention_subparser = self.enriched_attention_parser.add_subparsers(title='attention', help='Select attention function.',
                                                                            dest='attention_function', required=True)

        additive_attention_parser = attention_subparser.add_parser('additive',  help='Use additive attention.',
                                                              formatter_class=ArgumentDefaultsHelpFormatter)

        additive_attention_parser.set_defaults(attention_function='additive')


        additive_attention_parser.add_argument('--position-embedding-size', '-pes', type=int,
                                                    help='Size of the token distance embeddings.',
                                                    default=16)

        dot_product_attention_parser = attention_subparser.add_parser('dot_product', help='Use dot-product attention.',
                                                                   formatter_class=ArgumentDefaultsHelpFormatter)

        dot_product_attention_parser.set_defaults(attention_function='dot_product')

        dot_product_attention_parser.add_argument('--head-number', '-hn', type=int,
                                               help='Number of heads in the multilayer dot-product attention.',
                                               default=4)

        self.configure_global_features_parser(dot_product_attention_parser)
        self.configure_global_features_parser(additive_attention_parser)




    def configure_global_features_parser(self,
                                         parent_parser: argparse.ArgumentParser):

        global_features_parser = parent_parser.add_subparsers(title='global_features',
                                                                               help='Select the global features to use.',
                                                                               dest='global_feature', required=True)

        shortest_path_parser = global_features_parser.add_parser('shortest_path',
                                                                 help='Use the SDP as global feature.',
                                                                 formatter_class=ArgumentDefaultsHelpFormatter)

        shortest_path_parser.set_defaults(global_feature='shortest_path')

        entity_types_parser = global_features_parser.add_parser('entity_types',
                                                                help='Use entity types as global features.',
                                                                formatter_class=ArgumentDefaultsHelpFormatter)

        entity_types_parser.set_defaults(global_feature='entity_types')

        entity_types_parser.add_argument('--entity-embedding-size','-ees',
                                         help='Size for the entity embeddings',
                                         type=int,
                                         default=64)



    def parse_args(self) -> Dict[str, Union[int, str, float]] :

        # Retrieve the arguments as a dictionary
        args = vars(self.parser.parse_args())

        return args