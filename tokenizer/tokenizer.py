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
tokenizer class: it is responsible of retrieving the tokens IDs of each sample
"""

from transformers import RobertaTokenizer


class tokenizer(object):

    def __init__(self,
                 plm_model_path='roberta-base'):
        """
        Inits the tokenizer
        :param plm_model_path: path to the pretrained language model
        """

        # get roberta pre-trained tokenizer
        self.tokzer = RobertaTokenizer.from_pretrained(plm_model_path)

    def get_token_ids(self,
                      samples ):
        """
        Retrieve the token ids for each token and sample.
        :return: Array of ids
        """

        # Use the tokenizer to get the ids (of those words that have been previously tokenized)
        # option return_tesors: 'pt' (pytorch) because the model cannot work with raw python lists.
        # option padding: we need batched tensors with the same length.
        encoded_ids = self.tokzer( samples, is_split_into_words=True, return_tensors='pt', padding=True )

        return encoded_ids
