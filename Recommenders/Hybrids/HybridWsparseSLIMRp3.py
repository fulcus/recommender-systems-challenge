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

class HybridWsparseSLIMRp3(BaseItemSimilarityMatrixRecommender):
    """ HybridWsparseSLIMRp3
    Hybrid of two similarities S = S1*alpha + S2*(1-alpha)

    """

    RECOMMENDER_NAME = "HybridWsparseSLIMRp3"

    def __init__(self, URM_train, verbose=True):
        super(HybridWsparseSLIMRp3, self).__init__(URM_train, verbose=verbose)



        self.slim = SLIMElasticNetRecommender(
            URM_train)
        self.rp3 = RP3betaRecommender(URM_train)

        self.slim.load_model(output_root_path, file_name="slimelastic_urmall.zip")

        # self.slim.fit(topK=453, l1_ratio=0.00029920499017254754, alpha=0.10734084960757517)
        self.rp3.fit(topK=40, alpha=0.4208737801266599, beta=0.5251543657397256, normalize_similarity=True)

        Similarity_1 = self.slim.W_sparse
        Similarity_2 = self.rp3.W_sparse

        if Similarity_1.shape != Similarity_2.shape:
            raise ValueError(
                "ItemKNNSimilarityHybridRecommender: similarities have different size, S1 is {}, S2 is {}".format(
                    Similarity_1.shape, Similarity_2.shape
                ))

        # CSR is faster during evaluation
        self.Similarity_1 = check_matrix(Similarity_1.copy(), 'csr')
        self.Similarity_2 = check_matrix(Similarity_2.copy(), 'csr')

    def fit(self, topK=250, alpha=0.95):

        print('hyperparams: ' + str(topK), alpha)

        self.topK = topK
        self.alpha = alpha

        W_sparse = self.Similarity_1 * self.alpha + self.Similarity_2 * (1 - self.alpha)

        self.W_sparse = similarityMatrixTopK(W_sparse, k=self.topK)
        self.W_sparse = check_matrix(self.W_sparse, format='csr')
