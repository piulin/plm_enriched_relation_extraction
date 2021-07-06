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
Class dataset: defines the base methods for all kinds of datasets, e.g. ACE05 and TACRED
"""

class dataset(object):

    def get_tokens_of_samples(self):
        """
        Retrieves the tokens of each sample
        :return: List of tokens for each sample.
        """
        pass


    def __len__(self):
        """
        Gets the length of the dataset
        :return:
        """
        pass

    def __getitem__(self, i):
        """
        Retrieves the i-th sample of the dataset.
        :param i: index
        :return: i-th sample
        """
        pass

    def get_number_of_relations(self):
        """
        Retrieves the number of different relations in the dataset.
        :return:
        """
        pass