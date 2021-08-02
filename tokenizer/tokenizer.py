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
from typing import List, Tuple

"""
tokenizer class: it is responsible of retrieving the tokens IDs of each sample
"""

from transformers import RobertaTokenizer, BatchEncoding
from utils import constants
from log.teletype import teletype

class tokenizer(object):

    def __init__(self,
                 plm_model_path: str ='roberta-base'):
        """
        Inits the tokenizer
        :param plm_model_path: path to the pretrained language model, or one of the predefined strings to download one
        on the fly.
        """

        teletype.start_task('Configuring tokenizer', __name__)

        # get roberta pre-trained tokenizer
        self.tokzer: RobertaTokenizer = RobertaTokenizer.from_pretrained(plm_model_path)

        # define and add special tokens (Matching the blanks: distributional similarity for Relation Learning).
        special_tokens: dict = {
            'additional_special_tokens' : [ constants.E1S, constants.E1E, constants.E2S, constants.E2E ]
        }
        self.tokzer.add_special_tokens(special_tokens)

        teletype.finish_task(__name__)

    def get_token_ids(self,
                      samples: List[List[str]]) -> BatchEncoding:
        """
        Retrieve the token ids for each token and sample.
        :param samples: List of samples, each one containing only the tokens of that sample. `samples` is of shape
            [batch_size, sentence_length]
        :return: List of ids
        """

        # Use the tokenizer to get the ids (of those words that have been previously tokenized)
        # option return_tensors: 'pt' (pytorch) because the model cannot work with raw python lists.
        # option padding: we need batched tensors with the same length.
        # add_special_tokens: use new added special tokens
        encoded_ids: BatchEncoding = self.tokzer( samples,
                                   is_split_into_words=True,
                                   return_tensors='pt',
                                   padding=True,
                                   add_special_tokens=True )

        return encoded_ids

    def __len__(self) -> int:
        """
        Retrieves the vocabulary size of the pretrained tokenizer.
        :return:
        """
        return len(self.tokzer)

    def entity_tokens_ids(self) -> List[int]:
        """
        Retrieves the ids assigned to the tokens `constants.E1S`, `constants.E1E`, `constants.E2S`, and constants.E2E,
        respectively
        :return:
        """
        return self.tokzer.additional_special_tokens_ids

    def subtoken_mapping(self,
                         words: List[str]) -> Tuple[List[str], List[int]]:
        """
        Given a sequence of words or tokens `words`, this method retrieves the generated subtokens as well as the mapping
        to the original `words list`
        :param words: token/word list (size n)
        :return: subtoken list (size m >= n), and subtoken mapping list (size m)
        """

        tokens: List[str] = []
        tokens_map: List[int] = []

        # iterate words in list
        i: int
        word: str
        for i, word in enumerate(words):

            # tokenize
            _tokens: List[str] = self.tokzer.tokenize(word, add_prefix_space=True)

            # map subtokens with tokens/words
            token: str
            for token in _tokens:
                tokens.append(token)
                tokens_map.append(i)

        return tokens, tokens_map
