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
        self.data_dic [ 'token' ] = [ token.lower() for token in self.data_dic['token'] ]

