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
from typing import List, Tuple, Dict, Union

from torch import Tensor
from transformers import BatchEncoding

"""
tacred_emt class: a tacred dataset with support for Entity Token Markers
"""
import torch
from datasets.tacred.tacred import tacred
from datasets.tacred.sample import sample

class tacred_emt(tacred):

    def collate(self,
                data: List[Tuple[sample, int]]) ->  Dict[str,Union[BatchEncoding, Tensor, List[int]]]:
        """
        The collate function transforms the raw tokens into tokens IDs, puts the target Y list into tensors,
        and collects the start indices for mentioned entities of samples
        :param data: list of tuples (samples, y)
        :return: X and y data in tensors, and entity indices
        """

        # raw tokens (strings)
        tokens:List[List[str]] = []
        # raw gold labels
        y: List[int] = []
        # raw positions of E1S
        e1_indices: List[int] = []
        # raw positions of E2S
        e2_indices: List[int] = []

        # append to the lists declared above
        sample_id_tuple: Tuple[sample, int]
        for sample_id_tuple in data:

            sample: sample = sample_id_tuple[0]

            # tokens
            tokens.append(sample.emt_tokens)

            # gold labels
            y.append(sample_id_tuple[1])


        # retrieve tokens IDs and send them to the `device`. X[batch_size, padded_sentence_length]
        X: BatchEncoding = self.tokzer.get_token_ids(tokens).to(self.device)

        # from the tokenizer, get the ids of the emts for entity 1 and entity 2
        e1s_id: int
        e2s_id: int
        e1s_id, _, e2s_id, _ = self.tokzer.entity_tokens_ids()

        # loop over the the subtoken ids to retrieve the EMT positions
        i: int
        for i in range(X['input_ids'].shape[0]):
            # get subtoken ids of sentence
            # token_list[padded_sentence_length]
            token_list: Tensor = X['input_ids'][i,:]

            # retrieve the positions of EMTs
            e1s_position: int = (token_list == e1s_id).nonzero().item()
            e2s_position: int = (token_list == e2s_id).nonzero().item()

            # append them to the list
            e1_indices.append(e1s_position)
            e2_indices.append(e2s_position)



        # put targets into tensors and send them to the `device`
        y: Tensor = torch.tensor( y ).to(self.device)

        params = {
            'X': X,
            'y': y,
            'e1_indices': e1_indices,
            'e2_indices': e2_indices
        }



        return params

