#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 15/04/18

@author: Maurizio Ferrari Dacrema
"""
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from Recommenders.Recommender_utils import check_matrix, similarityMatrixTopK
from Recommenders.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender
from Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender


output_root_path = "./result_experiments/"

class BaseHybridSimilarity(BaseItemSimilarityMatrixRecommender):
    """
    Hybrid of two collaborative filtering models, obtained as weighted sum of their similarity matrices.
    W_sparse_hybrid = recommender_1.W_sparse * alpha + recommender_2.W_sparse * (1 - alpha)

    Note: recommender_1 and recommender_2 should already be fitted
    """

    RECOMMENDER_NAME = "BaseHybridSimilarity"

    def __init__(self, URM_train, recommender_1, recommender_2):
        super(BaseHybridSimilarity, self).__init__(URM_train, verbose=True)

        self.W_sparse = None
        self.topK = None
        self.alpha = None
        self.URM_train = check_matrix(URM_train.copy(), 'csr')

        self.recommender_1 = recommender_1
        self.recommender_2 = recommender_2

        similarity_1 = self.recommender_1.W_sparse
        similarity_2 = self.recommender_2.W_sparse

        if similarity_1.shape != similarity_2.shape:
            raise ValueError(
                "BaseHybridSimilarity: similarities have different size, S1 is {}, S2 is {}".format(
                    similarity_1.shape, similarity_2.shape
                ))

        # CSR is faster during evaluation
        self.similarity_1 = check_matrix(similarity_1.copy(), 'csr')
        self.similarity_2 = check_matrix(similarity_2.copy(), 'csr')

    def fit(self, topK=1199, alpha=0.961, beta=0.049):

        print('{} hyperparams: topK: {}; alpha: {}'.format(self.RECOMMENDER_NAME, str(topK), alpha))

        self.topK = topK
        self.alpha = alpha

        W_sparse = self.similarity_1 * self.alpha + self.similarity_2 * beta

        self.W_sparse = similarityMatrixTopK(W_sparse, k=self.topK)
        self.W_sparse = check_matrix(self.W_sparse, format='csr')
