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
import random

import torch
from datasets.dataset import dataset
import json
from os.path import join
from datasets.tacred.sample import sample
from tokenizer import mapper


class tacred(dataset):

    def __init__( self,
                  path_to_json,
                  tokzer,
                  device,
                  relation_mapper=None):
        """
        Loads a dataset into memory
        :param path_to_json: path to the TACRED folder
        :param tokzer: tokenizer
        :param device: torch device where the computation will take place
        :param relation_mapper: a mapper than translater relations to IDs and vice versa
        """

        # Get a mapper of relations to IDs.
        self.relation_mapper = mapper.seq_id() if relation_mapper is None else relation_mapper

        self.path_to_json = path_to_json
        self.device = device

        # Each one of the instances of the dataset and its corresponding relation ID
        self.samples, self.y = self.build_samples_from_json(self.path_to_json)

        self.tokzer = tokzer


    @staticmethod
    def build_file_paths(dataset_folder):
        """
        Builds the paths to the train, test, and development splits
        :param dataset_folder: base folder where the splits live.
        :return: Triple with the paths to the train, test, and development splits, respectively,
        """

        train_file = join( dataset_folder, 'train.json' )
        test_file = join( dataset_folder, 'test.json' )
        dev_file = join( dataset_folder, 'dev.json' )

        return train_file, dev_file, test_file


    def build_samples_from_json(self, file):
        """
        Parses the content of json splits.
        :param file: Path to the json split
        :return: array of samples.
        """

        # read json file
        with open(file, 'r') as fp:

            # parse json
            dump = json.load(fp)

            # instantiate array of samples
            samples = []

            # instantiate array of labels
            y = []

            for instance in dump:

                # Add sample
                samples.append(sample(instance))

                # retrieve relation ID
                y.append( self.relation_mapper.T2id(instance['relation']))

            return samples, y


    def get_tokens_of_samples(self):
        """
        Retrieves the raw tokens of the dataset.
        :return: list of raw tokens
        """
        return [ sample.tokens for sample in self.samples ]



    def __len__(self):
        """
        Retrieves the length of the dataset.
        :return:
        """

        return len( self.samples )

    def __getitem__(self, i):
        """
        Retrieves the i-th sample
        :param i: index
        :return: X and y
        """

        return self.samples[ i ], self.y [ i ]


    def collate(self, data):
        """
        The collate function transforms the raw tokens into tokens IDs, puts the target Y list into tensors,
        and collects the start indices for mentioned entities of samples
        :param data: list of tuples (samples, y)
        :return: X and y data in tensors, and entity indices
        """

        # raw tokens (strings)
        tokens = []
        # raw gold labels
        y = []
        # raw positions of E1S
        e1_indices = []
        # raw positions of E2S
        e2_indices = []

        # append to the lists declared above
        for sample_id_tuple in data:

            sample = sample_id_tuple[0]

            # tokens
            tokens.append(sample.tokens)

            # entity indices
            e1s, e2s = sample.emt_start_indices()

            # +1 to account for [CLS] token
            e1_indices.append(e1s + 1)
            e2_indices.append(e2s + 1)

            # gold labels
            y.append(sample_id_tuple[1])


        # retrieve tokens IDs and send them to the `device`
        X = self.tokzer.get_token_ids(tokens).to(self.device)

        # put targets into tensors and send them to the `device`
        y = torch.tensor( y ).to(self.device)


        return X, y, e1_indices, e2_indices

    def get_number_of_relations(self):
        """
        Retrieves the number of different relations in the dataset.
        :return:
        """
        return self.relation_mapper.no_entries()


    def get_relation_label_of_id(self, id):
        """
        Retrieves the label of relation with ID `id`
        :param id: relation ID
        :return: relation label
        """
        return self.relation_mapper.id2T(id)

    def save_subset(self,
                    path,
                    no_samples):
        """
        Saves a random subset of the dataset with `no_samples` samples to disk.
        :param path: save path
        :param no_samples: number of samples
        :return:
        """
        with open(path, 'w') as f:

            # selects random samples and puts the dictionaries into a nice array
            json_subset = [ s.data_dic for s in random.sample( self.samples, no_samples ) ]

            # save it into disk
            json.dump(json_subset, f)









