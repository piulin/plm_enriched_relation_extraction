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
from typing import Any, Dict

"""
seq_id class: it is responsible of mapping an object to an unique integer ID and vice versa.
"""

class seq_id(object):

    def __init__(self,
                 idx: int = 0,
                 lock_available: bool = False):
        """
        Sets up the dictionaries and indexes
        :param idx: starting index
        :param lock_available: adds an unknown entry '<UNKNOWN>' entry associated with index `idx`.
        """

        # Sets up dictionaries.
        self.freq_dict_: Dict[Any, int] = {}
        self.T_dict_: Dict[Any, int] = {}
        self.id_dict_: Dict[int, Any] = {}

        # Sets up starting index.
        self.idx_: int = idx

        # If lock, then introduce a mock work to the vocabulary
        if lock_available:
            self.T2id('<UNKNOWN>')


    def T2id(self,
             T: Any,
             lock: bool = False) -> int:
        """
        Assings to element T an new id if T is not in the dictionary, otherwise it retrieves its previously assigned id.
        :param T: element.
        :param lock: If `lock` is `True` and T is not in the dictionary, then the id of token '<UNKNOWN>' is
                returned.
        :return: id of element T.
        """

        # If T in dictionary
        if T in self.T_dict_.keys():

            # retrieve id
            id: int = self.T_dict_[T]

            # Update frequency of element
            self.freq_dict_[id] = self.freq_dict_[id] + 1
            return id

        else:

            # If lock, then return the id of '<UNKNOWN>'
            if lock:
                return self.T_dict_['<UNKNOWN>']

            # Otherwise, return the id of the new T
            self.T_dict_[T] = self.idx_
            self.freq_dict_[self.idx_] = 1
            self.id_dict_[self.idx_] = T
            self.idx_ += 1

            return self.idx_ - 1

    def id2T(self,
             id: int) -> Any:
        """
        Retrieves the object T given an ID.
        :param id: ID associated with object T.
        :return: object T associated with ID `id`
        """
        return self.id_dict_[id]

    def no_entries(self) -> int:
        """
        Retrieves the current indexing index
        :return:
        """
        return self.idx_