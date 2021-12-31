#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from sklearn.utils._testing import ignore_warnings

from Recommenders.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender
from Recommenders.Recommender_utils import check_matrix


class BaseHybridMultiRecommender(BaseItemSimilarityMatrixRecommender):
    """ BaseHybridMultiRecommender
    Hybrid of N prediction scores R
    """

    RECOMMENDER_NAME = "BaseHybridMultiRecommender"

    def __init__(self, URM_train, recommender_array, number_of_recommenders):
        super(BaseHybridMultiRecommender, self).__init__(URM_train)
        self.number_of_recommenders = number_of_recommenders
        self.URM_train = check_matrix(URM_train.copy(), 'csr')
        self.recommender_array = recommender_array
        print('number of recommenders:', self.number_of_recommenders)

    @ignore_warnings()
    def fit(self, weight_array):
        self.weight_array = weight_array

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        item_weights = []
        for i in range(self.number_of_recommenders):
            item_weights.append(self.recommender_array[i]._compute_item_score(user_id_array))

        weighted_matrices = [a * b for a, b in zip(item_weights, self.weight_array)]
        item_weights = sum(weighted_matrices)

        return item_weights
