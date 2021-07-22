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
from utils import constants


class tokenizer(object):

    def __init__(self,
                 plm_model_path='roberta-base'):
        """
        Inits the tokenizer
        :param plm_model_path: path to the pretrained language model
        """

        # get roberta pre-trained tokenizer
        self.tokzer = RobertaTokenizer.from_pretrained(plm_model_path)

        # define and add special tokens (Matching the blanks: distributional similarity for Relation Learning).
        special_tokens = {
            'additional_special_tokens' : [ constants.E1S, constants.E1E, constants.E2S, constants.E2E ]
        }

        self.tokzer.add_special_tokens(special_tokens)

    def get_token_ids(self,
                      samples ):
        """
        Retrieve the token ids for each token and sample.
        :return: Array of ids
        """

        # Use the tokenizer to get the ids (of those words that have been previously tokenized)
        # option return_tensors: 'pt' (pytorch) because the model cannot work with raw python lists.
        # option padding: we need batched tensors with the same length.
        # add_special_tokens: use new added special tokens
        encoded_ids = self.tokzer( samples,
                                   is_split_into_words=True,
                                   return_tensors='pt',
                                   padding=True,
                                   add_special_tokens=True )

        return encoded_ids

    def __len__(self):
        """
        Retrieves the vocabulary size of the pretrained tokenizer.
        :return:
        """
        return len(self.tokzer)

    def entity_tokens_ids(self):
        """
        Retrieves the ids assigned to the tokens `constants.E1S`, `constants.E1E`, `constants.E2S`, and constants.E2E,ç
        respectively
        :return:
        """
        return self.tokzer.additional_special_tokens_ids

    def subtoken_mapping(self, words):
        """
        Given a sequence of words or tokens `words`, this method retrieves the generated subtokens as well as the mapping
        to the original `words list`
        :param words: token or workd list
        :return: subtoken list, and subtoken mapping
        """

        tokens = []
        tokens_map = []

        # iterate words in list
        for i, word in enumerate(words):

            # tokenize
            _tokens = self.tokzer.tokenize(word, add_prefix_space=True)

            # map subtokens with tokens/words
            for token in _tokens:
                tokens.append(token)
                tokens_map.append(i)

        return tokens, tokens_map
