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
tacred_enriched class: a tacred dataset with support for enriched attention
"""
import torch
from datasets.tacred.tacred import tacred
from utils.constants import sdp_flag_padding_index
from datasets.tacred.sample import sample
class tacred_enriched(tacred):

    def collate(self,
                data: List[Tuple[sample, int]]) -> Dict[str,Union[BatchEncoding, Tensor]]:
        """
        The collate function transforms the raw tokens into tokens IDs, puts the target Y list into tensors,
        and collects the features necessary for enriched attention
        :param data: list of tuples (samples, y)
        :return: X and y[batch_size] data, as well as dependency distances to
        entity 1 DE1[batch_size, padded_sentence_length] and entity 2 DE2[batch_size, padded_sentence_length],
        SDP flags SDP_FLAG[batch_size, padded_sentence_length],
        SDP itself, token distances to entity 1 PO[batch_size, padded_sentence_length] and
        entity 2 PS[batch_size, padded_sentence_length], and the padding mask MASK[batch_size, padded_sentence_length],
         respectively.
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

        # mask of padding elements
        mask: List[List[bool]] = []

        # entity types
        entity_types: List[List[int]] = []

        # length of the largest sentence
        largest_stc_length: int = 0

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
            if len(m) > largest_stc_length:
                largest_stc_length = len(m)

            # enriched attention attributes (defined above) extended to match subtokenization
            de1.append( self.extend_idxs( sample.de1, m ) )
            de2.append( self.extend_idxs( sample.de2, m ) )
            sdp_flag.append( self.extend_idxs( sample.sdp_flag, m ) )
            sdp.append(  sample.sdp )
            po.append( self.extend_idxs( sample.po, m ) )
            ps.append( self.extend_idxs( sample.ps, m ) )

            # entity types
            entity_types.append( sample.entity_types )

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

            self.add_padding(de1[i], largest_stc_length, dep_padding_idx)
            self.add_padding(de2[i], largest_stc_length, dep_padding_idx)
            self.add_padding(sdp_flag[i], largest_stc_length, sdp_flag_padding_index)
            self.add_padding(po[i], largest_stc_length, tok_padding_idx)
            self.add_padding(ps[i], largest_stc_length, tok_padding_idx)

            # retrieve original seq length
            seq_len: int = len(data[i][0].tokens)
            # create padding mask
            mask.append(self.padding_mask( seq_len, largest_stc_length ))

        # put enriched attention-related features into the `device`
        DE1: Tensor = torch.tensor(de1).to(self.device) # DE1[batch_size, padded_sentence_length]
        DE2: Tensor = torch.tensor(de2).to(self.device) # DE2[batch_size, padded_sentence_length]
        SDP_FLAG: Tensor = torch.tensor(sdp_flag).to(self.device) # SDP_FLAG[batch_size, padded_sentence_length]
        PO: Tensor = torch.tensor(po).to(self.device) # PO[batch_size, padded_sentence_length]
        PS: Tensor = torch.tensor(ps).to(self.device) # PS[batch_size, padded_sentence_length]
        MASK: Tensor = torch.tensor(mask).to(self.device) # MASK[batch_size, padded_sentence_length]
        ET: Tensor = torch.tensor(entity_types).to(self.device) # ET[batch_size, padded_sentence_length]

        params = {
            'X': X,
            'y': y,
            'ps': PS,
            'po': PO,
            'de1': DE1,
            'de2': DE2,
            'f': SDP_FLAG,
            'sdp': SDP,
            'mask': MASK,
            'entity_types': ET
        }

        return params

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

    def padding_mask(self,
                     seq_length: int,
                     total_length: int) -> List[bool]:
        """
        Creates a padding mask list such that the first `seq_length` elements of that list are `False`, and the next
        `total_length - seq_length` elements are `True`, i.e., positions of not padded elements are `False`, whereas positions
        of padded elements are `True`
        :param seq_length: sequence length (number of true elements)
        :param total_length: total length accounting for padding
        :return: mask list of size `total_length`
        """
        outlist: List[bool] = []

        outlist.extend([False]*seq_length)
        outlist.extend([True]*(total_length-seq_length))

        return outlist


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

