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
from typing import List, Tuple, Any

"""
Class dataset: defines the methods that a dataset needs to implement in order to perform training
"""

class dataset(object):

    def get_tokens_of_samples(self) ->  List[List[str]]:
        """
        Retrieves the tokens of each sample
        :return: List of tokens for each sample.
        """
        pass


    def __len__(self) -> int:
        """
        Gets the length of the dataset
        :return:
        """
        pass

    def __getitem__(self,
                    i: int) -> Tuple[Any, int]:
        """
        Retrieves the i-th sample of the dataset, and its corresponding label.
        :param i: index
        :return: i-th sample and label
        """
        pass

    def collate(self,
                data:  List[Tuple[Any, int]]) -> Any:
        """
        Collates samples.
        :param data:  list of tuples (samples, y)
        :return:
        """

    def get_number_of_relations(self) -> int:
        """
        Retrieves the number of different relations in the dataset.
        :return:
        """
        pass

    def get_relation_label_of_id(self,
                                 id: int) -> str:
        """
        Retrieves the label of relation with ID `id`
        :param id: relation ID
        :return: relation label
        """
        pass

    def no_relation_label(self) -> int:
        """
        Returns the ID of the no-relation type.
        :return:
        """
        pass

    def highest_token_entity_distance(self) -> int:
        """
        Retrieves the highest distance from a token to an entity in the whole dataset
        :return:
        """
        pass

    def highest_dependency_entity_distance(self) -> int:
        """
        Retrieves the maximum distance of an entity to a token in the dependency parse for the whole dataset
        :return:
        """
        pass

    def get_number_of_entity_types(self) -> int:
        """
        Retrieves the number of different entity types in the dataset
        :return:
        """
        pass