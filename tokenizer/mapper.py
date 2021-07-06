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
seq_id class: it is responsible of mapping an object to an unique integer ID and vice versa.
"""

class seq_id(object):

    def __init__(self, idx=0, lock_available=False):
        """
        Sets up the dictionaries and indexes required
        :param idx: starting index
        :param lock_available: adds an unknow entry '<UNKNOWN>' entry associated with index `idx`.
        """

        # Sets up dictionaries.
        self.freq_dict_ = {}
        self.T_dict_ = {}
        self.id_dict_ = {}

        # Sets up starting index.
        self.idx_ = idx

        # If lock, then introduce a mock work to the vocabulary
        if lock_available:
            self.T2id('<UNKNOWN>')


    def T2id(self, T, lock=False):
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
            id = self.T_dict_[T]

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

    def id2T(self, id):
        """
        Retrieves the object T given an ID.
        :param id: ID associated with object T.
        :return: object T associated with ID `id`
        """
        return self.id_dict_[id]

    def no_entries(self):
        """
        Retrieves the current indexing index
        :return:
        """
        return self.idx_