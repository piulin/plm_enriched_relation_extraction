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
Class sample: stores each one of the samples of the TACRED dataset.
"""

from utils import constants


class sample (object) :

    def __init__(self, data_dic ):

        self.data_dic = data_dic

        self.prepare()

    def prepare(self):
        """
        Preprocess the sample, if needed
        :return:
        """

        # lowercase all tokens
        # self.data_dic [ 'token' ] = [ token.lower() for token in self.data_dic['token'] ]

        # Augment tokens
        self.augment_EMT()



    def augment_EMT(self):
        """
        Augment tokens with reserved word to mark the begin and end of each entity mention.
        EMT stands for Entity Marker Tokens
        :return:
        """

        # Retrieve entity mentions
        obj_start =   self.data_dic['obj_start']
        obj_end =     self.data_dic['obj_end']

        subj_start =  self.data_dic['subj_start']
        subj_end =    self.data_dic['subj_end']

        # copy token list
        emt = list(self.data_dic['token'])

        # The indices are order-dependent
        if obj_start < subj_start:

            # Insert special tokens
            emt.insert(obj_start, constants.E1S)
            emt.insert(obj_end + 2, constants.E1E)

            emt.insert(subj_start + 2, constants.E2S)
            emt.insert(subj_end + 4, constants.E2E)

            # Save positions of start entity tokens
            self.data_dic['EMT_e1s'] = obj_start
            self.data_dic['EMT_e2s'] = subj_start + 2

        else:

            # Insert special tokens
            emt.insert(subj_start, constants.E1S)
            emt.insert(subj_end + 2, constants.E1E)

            emt.insert(obj_start + 2, constants.E2S)
            emt.insert(obj_end + 4, constants.E2E)

            # Save positions of start entity tokens
            self.data_dic['EMT_e1s'] = subj_start
            self.data_dic['EMT_e2s'] = obj_start + 2


        # append it to the dict
        self.data_dic ['token_EMT'] = emt

    @property
    def tokens(self):
        """
        Retrieves tokens of sample
        :return:
        """
        return self.data_dic['token_EMT']

    def emt_start_indices(self):
        """
        Retrieves the start indices of EMT augmented tokens `constants.E1S` and `constants.E2S`
        :return: 
        """

        # +2 accounts for the spaces generated after inserting `constants.E1S `and `constants.E1E`.
        return self.data_dic['EMT_e1s'], self.data_dic['EMT_e2s']


