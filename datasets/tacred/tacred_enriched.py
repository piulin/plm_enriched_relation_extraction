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

from torch import Tensor
from transformers import BatchEncoding

"""
tacred_enriched class: a tacred dataset with support for enriched attention
"""
import torch
from datasets.tacred.tacred import tacred
from utils.constants import sdp_flag_padding_index
from datasets.tacred.sample import sample
class tacred_enriched(tacred):

    def collate(self,
                data: List[Tuple[sample, int]]) -> \
            Tuple[BatchEncoding, Tensor, Tensor, Tensor, Tensor, BatchEncoding, Tensor,
                  Tensor]:
        """
        The collate function transforms the raw tokens into tokens IDs, puts the target Y list into tensors,
        and collects the features necessary for enriched attention
        :param data: list of tuples (samples, y)
        :return: X and y[batch_size] data, as well as dependency distances to
        entity 1 DE1[batch_size, padded_sentence_length] and entity 2 DE2[batch_size, padded_sentence_length],
        SDP flags SDP_FLAG[batch_size, padded_sentence_length],
        SDP itself, and
        token distances to entity 1 PO[batch_size, padded_sentence_length] and
        entity 2 PS[batch_size, padded_sentence_length], respectively
        """

        # raw tokens (strings)
        tokens: List[List[str]] = []
        # raw gold labels
        y: List[int] = []

        # dependency distances to entity 1
        de1: List[List[int]] = []
        # dependency distances to entity 2
        de2: List[List[int]] = []
        # shortest dependency path flags
        sdp_flag: List[List[int]] = []
        # shortest dependency path
        sdp: List[List[str]] = []

        # token distances to entity 1
        ps: List[List[int]] = []
        # token distances to entity 2
        po: List[List[int]] = []

        # length of the largest sentence
        length: int = 0

        # append to the lists declared above
        sample_id_tuple: Tuple[sample,int]
        for sample_id_tuple in data:

            sample: sample = sample_id_tuple[0]

            # tokens
            tokens.append(sample.tokens)

            # subtoken mapping to token-level words
            m: List[int]
            _, m = self.tokzer.subtoken_mapping(sample.tokens)

            # gold labels
            y.append(sample_id_tuple[1])

            # update largest length (subtoken level)
            if len(m) > length:
                length = len(m)

            # enriched attention attributes (defined above) extended to match subtokenization
            de1.append( self.extend_idxs( sample.de1, m ) )
            de2.append( self.extend_idxs( sample.de2, m ) )
            sdp_flag.append( self.extend_idxs( sample.sdp_flag, m ) )
            sdp.append(  sample.sdp )
            po.append( self.extend_idxs( sample.po, m ) )
            ps.append( self.extend_idxs( sample.ps, m ) )

        # retrieve tokens IDs and send them to the `device` for both tokens and SDP
        X: BatchEncoding = self.tokzer.get_token_ids(tokens).to(self.device)
        SDP: BatchEncoding = self.tokzer.get_token_ids(sdp).to(self.device)

        # put targets into tensors and send them to the `device`. y[batch_size]
        y: Tensor = torch.tensor( y ).to(self.device)

        # modify the enriched features to account for the padding
        # +1 defines the padding idx
        dep_padding_idx: int = self.highest_dependency_distance+1
        tok_padding_idx: int = self.highest_token_distance+1

        i: int
        for i in range(len(de1)):

            self.add_padding(de1[i], length, dep_padding_idx)
            self.add_padding(de2[i], length, dep_padding_idx)
            self.add_padding(sdp_flag[i], length, sdp_flag_padding_index)
            self.add_padding(po[i], length, tok_padding_idx)
            self.add_padding(ps[i], length, tok_padding_idx)

        # put enriched attention-related features into the `device`
        DE1: Tensor = torch.tensor(de1).to(self.device) # DE1[batch_size, padded_sentence_length]
        DE2: Tensor = torch.tensor(de2).to(self.device) # DE2[batch_size, padded_sentence_length]
        SDP_FLAG: Tensor = torch.tensor(sdp_flag).to(self.device) # SDP_FLAG[batch_size, padded_sentence_length]
        PO: Tensor = torch.tensor(po).to(self.device) # PO[batch_size, padded_sentence_length]
        PS: Tensor = torch.tensor(ps).to(self.device) # PS[batch_size, padded_sentence_length]


        return X, y, DE1, DE2, SDP_FLAG, SDP, PO, PS

    def add_padding(self,
                    v: List[int],
                    max_length: int,
                    padding_idx: int) -> None:
        """
        Adds padding `padding_idx` to a list `v` up to `max_length` positions
        :param v: vector to pad
        :param max_length: padding length
        :param padding_idx: padding value
        :return:
        """
        v.extend([padding_idx] * (max_length - len(v)))


    def extend_idxs(self,
                    target: List[int],
                    indices: List[int]) -> List[int]:
        """
        Extend `target` list by including only those `target` elements listed in `indices`
        :param target: list to be extended
        :param indices: list of indices indicating those elements of the extended list
        :return: new list
        """

        return [ target[i] for i in indices ]

