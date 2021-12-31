#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import scipy.sparse as sps
import numpy as np
from sklearn.utils._testing import ignore_warnings

from Recommenders.EASE_R.EASE_R_Recommender import EASE_R_Recommender
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from Recommenders.Hybrids.BaseHybridMultiRecommender import BaseHybridMultiRecommender
from Recommenders.MatrixFactorization.IALSRecommender_implicit import IALSRecommender_implicit
from Recommenders.MatrixFactorization.PureSVDRecommender import PureSVDRecommender
from Recommenders.Recommender_utils import check_matrix
from Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender
from reader import load_icm

output_root_path = "./result_experiments/"


class MultiRecommender(BaseHybridMultiRecommender):
    """ MultiRecommender
    Hybrid of N prediction scores R
    """

    RECOMMENDER_NAME = "MultiRecommender"

    def __init__(self, URM_train):
        slim = SLIMElasticNetRecommender(URM_train)
        slim.load_model(output_root_path, file_name="slim742.zip")
        # self.slim.fit(topK=453, l1_ratio=0.00029920499017254754, alpha=0.10734084960757517)

        rp3 = RP3betaRecommender(URM_train)
        rp3.fit(topK=40, alpha=0.4208737801266599, beta=0.5251543657397256, normalize_similarity=True)

        ICM_event = load_icm("data_ICM_event.csv", weight=1)
        ICM_event = check_matrix(ICM_event.T, 'csr', dtype=np.float32)
        URM_ICM_event = sps.vstack((URM_train, ICM_event), format='csr', dtype=np.float32)

        ease_r = EASE_R_Recommender(URM_ICM_event)
        ease_r.fit()

        ials = IALSRecommender_implicit(URM_train)
        ials.fit()

        pure_svd = PureSVDRecommender(URM_train)
        pure_svd.fit()

        super(MultiRecommender, self).__init__(URM_train, [slim, rp3, ease_r, ials, pure_svd], 5)

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
