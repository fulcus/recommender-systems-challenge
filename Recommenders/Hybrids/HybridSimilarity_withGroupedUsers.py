#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 15/04/18

@author: Maurizio Ferrari Dacrema
"""
import numpy as np
from tqdm import tqdm

from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from Recommenders.Hybrids.BaseHybridSimilarity import BaseHybridSimilarity
from Recommenders.Recommender_utils import check_matrix, similarityMatrixTopK
from Recommenders.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender
from Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender

import scipy.sparse as sps


output_root_path = "./result_experiments/"

class HybridSimilarity_withGroupedusers(BaseItemSimilarityMatrixRecommender):
    """ HybridSimilarity_SLIM_Rp3
    Hybrid of two similarities S = S1*alpha + S2*(1-alpha)
    """

    RECOMMENDER_NAME = "HybridSimilarity_withGroupedusers"

    def __init__(self, URM_train):
        super(HybridSimilarity_withGroupedusers, self).__init__(URM_train)

        slim = SLIMElasticNetRecommender(URM_train)
        rp3_g0 = RP3betaRecommender(URM_train)
        rp3_g1 = RP3betaRecommender(URM_train)

        slim.load_model(output_root_path, file_name="slim_splitforeval742.zip")
        # self.slim.fit(topK=453, l1_ratio=0.00029920499017254754, alpha=0.10734084960757517)
        rp3_g0.fit( topK=725, alpha= 0.8, beta= 0.6533987658966547, normalize_similarity= True)
        rp3_g1.fit(topK= 42, alpha=0.0, beta=0.576800562870638, normalize_similarity=True)
        # rp3_g1.fit(topK=40, alpha=0.4208737801266599, beta=0.5251543657397256, normalize_similarity=True)
        self.W_sparse = None
        self.topK = None
        self.alpha = None
        self.URM_train = check_matrix(URM_train.copy(), 'csr')

        self.recommender_1 = slim
        self.recommender_2 = rp3_g0
        self.recommender_3 = rp3_g1

        similarity_1 = self.recommender_1.W_sparse
        similarity_2 = self.recommender_2.W_sparse
        similarity_3 = self.recommender_3.W_sparse

        if similarity_1.shape != similarity_2.shape:
            raise ValueError(
                "BaseHybridSimilarity: similarities have different size, S1 is {}, S2 is {}".format(
                    similarity_1.shape, similarity_2.shape
                ))

        if similarity_1.shape != similarity_3.shape:
            raise ValueError(
                "BaseHybridSimilarity: similarities have different size, S1 is {}, S2 is {}".format(
                    similarity_1.shape, similarity_2.shape
                ))

        # CSR is faster during evaluation
        self.similarity_1 = check_matrix(similarity_1.copy(), 'csr')
        self.similarity_2 = check_matrix(similarity_2.copy(), 'csr')
        self.similarity_3 = check_matrix(similarity_3.copy(), 'csr')

        ##### groups

        group_id=0

        profile_length = np.ediff1d(sps.csr_matrix(URM_train).indptr)
        print("profile", profile_length, profile_length.shape)

        block_size = int(len(profile_length) * 0.5)
        print("block_size", block_size)

        sorted_users = np.argsort(profile_length)
        print("sorted users", sorted_users)

        start_pos = group_id * block_size
        end_pos = min((group_id + 1) * block_size, len(profile_length))

        users_in_group = sorted_users[start_pos:end_pos]

        users_in_group_p_len = profile_length[users_in_group]

        print("Group {}, #users in group {}, average p.len {:.2f}, median {}, min {}, max {}".format(
            group_id,
            users_in_group.shape[0],
            users_in_group_p_len.mean(),
            np.median(users_in_group_p_len),
            users_in_group_p_len.min(),
            users_in_group_p_len.max()))

        users_not_in_group_flag = np.isin(sorted_users, users_in_group, invert=True)
        self.users_not_in_group0 = sorted_users[users_not_in_group_flag]

        # print("users group 1", self.users_not_in_group0)

        ####

    def fit(self, topK=250, alpha=0.95):
        print('{} hyperparams: topK: {}; alpha: {}'.format(self.RECOMMENDER_NAME, str(topK), alpha))

        self.topK = topK
        self.alpha = alpha

        W_sparse_0 = self.similarity_1 * self.alpha + self.similarity_2 * (1 - self.alpha)
        W_sparse_1 = self.similarity_1 * self.alpha + self.similarity_3 * (1 - self.alpha)

        self.W_sparse_0 = similarityMatrixTopK(W_sparse_0, k=self.topK)
        self.W_sparse_0 = check_matrix(self.W_sparse_0, format='csr')

        self.W_sparse_1 = similarityMatrixTopK(W_sparse_1, k=self.topK)
        self.W_sparse_1 = check_matrix(self.W_sparse_1, format='csr')

        self.W_sparse = self.W_sparse_0




    def _compute_item_score(self, user_id_array, items_to_compute=None):
        """
        URM_train and W_sparse must have the same format, CSR
        :param user_id_array:
        :param items_to_compute:
        :return:
        """
        # print("users target ", user_id_array)

        self._check_format()



        '''users_not_in_group_flag = np.isin(user_id_array, self.users_not_in_group0 , invert=True)
        user_id_array_1 = user_id_array[users_not_in_group_flag]
        user_id_array_0 = user_id_array[]

        for user_id in user_id_array:
            if user_id in self.users_not_in_group0:
                user_id_array_1.add(user_id)
            else:
                user_id_array_0.add(user_id)

        user_profile_array_0 = self.URM_train[user_id_array_0]
        user_profile_array_1 = self.URM_train[user_id_array_1]

        if items_to_compute is not None:
            item_scores = - np.ones((len(user_id_array), self.URM_train.shape[1]), dtype=np.float32) * np.inf
            item_scores_0 = user_profile_array_0.dot(self.W_sparse).toarray()

            item_scores_1 = user_profile_array_1.dot(self.W_sparse_1).toarray()

            item_scores_all = np.concatenate((item_scores_0,item_scores_1), axis=1)

            item_scores[:, items_to_compute] = item_scores_all[:, items_to_compute]

        else:
            item_scores_0 = user_profile_array_0.dot(self.W_sparse).toarray()
            item_scores_1 = user_profile_array_1.dot(self.W_sparse_1).toarray()

            item_scores = np.concatenate((item_scores_0,item_scores_1), axis=1)

        return item_scores'''

        item_scores = np.empty([len(user_id_array), 18059])

        for i in tqdm(range(len(user_id_array))):

            interactions = len(self.URM_train[user_id_array[i], :].indices)

            if interactions < 300:
                self.W_sparse= self.W_sparse_0
                # w = BaseItemSimilarityMatrixRecommender._compute_item_score(user_id_array[i],items_to_compute)
                # item_scores[i, :] = w

            else:
                self.W_sparse = self.W_sparse_1
                # w = BaseItemSimilarityMatrixRecommender._compute_item_score(user_id_array[i], items_to_compute)

                # item_scores[i, :] = w

            user_profile_array = self.URM_train[user_id_array[i]]

            if items_to_compute is not None:
                item_scores = - np.ones((len(user_id_array[i]), self.URM_train.shape[1]), dtype=np.float32) * np.inf
                item_scores_all = user_profile_array.dot(self.W_sparse).toarray()
                item_scores[i, :] = item_scores_all
            else:
                item_scores[i, :] = user_profile_array.dot(self.W_sparse).toarray()

        return item_scores
