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
import copy

"""
Class sample: stores each one of the samples of the TACRED dataset.
"""

from utils import constants
import datasets.tacred.dependency_parse as dp


class sample(object):

    def __init__(self, data_dic):

        # sample attributes
        self.data_dic = data_dic

        # stores the highest distance from a token to an entity, token-wise and dependency-wise
        self.highest_entity_distance = 0
        self.highest_dependency_distance = 0

        self.prepare()

    def prepare(self):
        """
        Preprocess the sample, if needed
        :return:
        """

        # lowercase all tokens
        # self.data_dic [ 'token' ] = [ token.lower() for token in self.data_dic['token'] ]

        # Augment tokens, including Entity Marker Tokens (EMT)
        self.augment_EMT()

        # get dependency distance-related features (e.g. SDP)
        self.dependency_distance()

        # get token distances to entities
        self.entities_distance()

    def entities_distance(self):
        """
        Computes the distances to the entities for each token
        :return:
        """

        # Get entity indexes
        ss, se = self.data_dic['subj_start'], self.data_dic['subj_end']
        os, oe = self.data_dic['obj_start'], self.data_dic['obj_end']

        # Retrieve distances to entities for each token
        distances_to_e1 = self.distance_to_entity(ss, se)
        distances_to_e2 = self.distance_to_entity(os, oe)

        # save data
        self.data_dic['ps'] = distances_to_e1
        self.data_dic['po'] = distances_to_e2

    def distance_to_entity(self,
                           begin,
                           end):
        """
        Calculates the distance of all tokens with respect to the entity starting at index `begin` and ending at index
        `end`.
        :param begin: entity beginning index
        :param end: entity ending index
        :return: distances of all tokens to entity
        """

        # define output list
        distances = []

        # iterate all tokens
        for i in range(len(self.data_dic['token'])):

            distance = 0

            # if token is part of the entity, then the distance is 0
            if begin <= i <= end:
                distance = 0
            # otherwise, choose the shortest distance to the entity
            elif abs(i - begin) < abs(i - end):
                distance = abs(i - begin)
            else:
                distance = abs(i - end)

            # append distance of token to list
            distances.append(distance)

            # update highest distance value
            if self.highest_entity_distance < distance:
                self.highest_entity_distance = distance

        return distances

    def augment_EMT(self):
        """
        Augment tokens with reserved word to mark the begin and end of each entity mention.
        EMT stands for Entity Marker Tokens
        :return:
        """

        # Retrieve entity mentions
        obj_start = self.data_dic['obj_start']
        obj_end = self.data_dic['obj_end']

        subj_start = self.data_dic['subj_start']
        subj_end = self.data_dic['subj_end']

        # copy token list
        emt = list(self.data_dic['token'])

        # The indices are order-dependent
        if obj_start < subj_start:

            # Insert special tokens
            emt.insert(obj_start, constants.E1S)
            # +2 accounts for the spaces generated after inserting `constants.E1S `and `constants.E1E`.
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
        self.data_dic['token_EMT'] = emt

    def dependency_distance(self):
        """
        For each token, this function calculates the distance to the two query entities IN the dependency parse tree
        :return:
        """

        # convert dependency heads into a dictionary
        tree = dp.read_dependency_tree(self.data_dic)

        # retrieve entities idxs and tokens
        ss, se = self.data_dic['subj_start'], self.data_dic['subj_end']
        os, oe = self.data_dic['obj_start'], self.data_dic['obj_end']
        tokens_orig = copy.deepcopy(self.data_dic['token'])

        # get distances to e1, to 2, the shortest path, and the flag for those tokens in the shortest path
        distances_to_e1, \
        distances_to_e2, \
        shortest_path, \
        flag_on_shortest_path, \
        self.highest_dependency_distance = dp.get_info_from_dependency((ss, se),
                                                                  (os, oe),
                                                                  tree,
                                                                  tokens_orig)
        # save data
        self.data_dic['de1'] = distances_to_e1
        self.data_dic['de2'] = distances_to_e2
        self.data_dic['sdp'] = shortest_path
        self.data_dic['sdp_flag'] = flag_on_shortest_path

    @property
    def emt_tokens(self):
        """
        Retrieves tokens of sample
        :return:
        """
        return self.data_dic['token_EMT']

    @property
    def tokens(self):
        """
        Retrieves tokens of sample
        :return:
        """
        return self.data_dic['token']

    @property
    def de1(self):
        """
        Retrieves dependency distances to entity 1
        :return:
        """
        return self.data_dic['de1']

    @property
    def de2(self):
        """
        Retrieves dependency distances to entity 2
        :return:
        """
        return self.data_dic['de2']

    @property
    def sdp(self):
        """
        Retrieves shortest dependency path
        :return:
        """
        return self.data_dic['sdp']

    @property
    def sdp_flag(self):
        """
        Retrieves shortest dependency path flags for each token. If a token is in the SDP, then its value is 1, otherwise
        -1.
        :return:
        """
        return self.data_dic['sdp_flag']

    @property
    def ps(self):
        """
        Retrieves token distances to entity 1
        :return:
        """
        return self.data_dic['ps']

    @property
    def po(self):
        """
        Retrieves token distances to entity 2
        :return:
        """
        return self.data_dic['po']

    @property
    def highest_token_entity_distance(self):
        """
        Retrieves the maximum distance of an entity to a token in the token list
        :return:
        """
        return self.highest_entity_distance

    @property
    def highest_dependency_entity_distance(self):
        """
        Retrieves the maximum distance of an entity to a token in the dependency parse
        :return:
        """
        return self.highest_dependency_distance

    def emt_start_indices(self):
        """
        Retrieves the start indices of EMT augmented tokens `constants.E1S` and `constants.E2S`
        :return: 
        """
        return self.data_dic['EMT_e1s'], self.data_dic['EMT_e2s']
