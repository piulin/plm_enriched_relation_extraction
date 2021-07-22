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
tacred_enriched class: a tacred dataset with support for enriched attention
"""
import torch
from datasets.tacred.tacred import tacred
from utils.constants import sdp_flag_padding_index

class tacred_enriched(tacred):

    def collate(self, data):
        """
        The collate function transforms the raw tokens into tokens IDs, puts the target Y list into tensors,
        and collects the features necessary for enriched attention
        :param data: list of tuples (samples, y)
        :return: X and y data, as well as dependency distances to entity 1 and 2, SDP flags, SDP itself, and
        token distances to entity 1 and 2, respectively
        """

        # raw tokens (strings)
        tokens = []
        # raw gold labels
        y = []

        # dependency distances to entity 1
        de1 = []
        # dependency distances to entity 2
        de2 = []
        # shortest dependency path flags
        sdp_flag = []
        # shortest dependency path
        sdp = []

        # token distances to entity 1
        ps = []
        # token distances to entity 2
        po = []

        # length of the largest sentence
        length = 0

        # append to the lists declared above
        for sample_id_tuple in data:

            sample = sample_id_tuple[0]

            # tokens
            tokens.append(sample.tokens)


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
        X = self.tokzer.get_token_ids(tokens).to(self.device)
        SDP = self.tokzer.get_token_ids(sdp).to(self.device)

        # put targets into tensors and send them to the `device`
        y = torch.tensor( y ).to(self.device)

        # modify the enriched features to account for the padding
        # +1 defines the padding idx
        dep_padding_idx = self.highest_dependency_distance+1
        tok_padding_idx = self.highest_token_distance+1
        for i in range(len(de1)):

            self.add_padding(de1[i], length, dep_padding_idx)
            self.add_padding(de2[i], length, dep_padding_idx)
            self.add_padding(sdp_flag[i], length, sdp_flag_padding_index)
            self.add_padding(po[i], length, tok_padding_idx)
            self.add_padding(ps[i], length, tok_padding_idx)

        # put enriched attention-related features into the `device`
        DE1 = torch.tensor(de1).to(self.device)
        DE2 = torch.tensor(de2).to(self.device)
        SDP_FLAG = torch.tensor(sdp_flag).to(self.device)
        PO = torch.tensor(po).to(self.device)
        PS = torch.tensor(ps).to(self.device)


        return X, y, DE1, DE2, SDP_FLAG, SDP, PO, PS

    def add_padding(self, v, max_length, padding_idx):
        """
        Adds padding `padding_idx` to a vector `v` up to `max_length` positions
        :param v: vector to pad
        :param max_length: padding length
        :param padding_idx: padding value
        :return:
        """
        v.extend([padding_idx] * (max_length - len(v)))


    def extend_idxs(self,
                    target,
                    indices):
        """
        Extend `target` list by including only those `target` elements listed in `indices`
        :param target: list to be extended
        :param indices: list of indices indicating those elements of the extended list
        :return: new list
        """

        return [ target[i] for i in indices ]

