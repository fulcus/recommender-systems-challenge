#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from sklearn.utils._testing import ignore_warnings

from Recommenders.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender
from Recommenders.EASE_R.EASE_R_Recommender import EASE_R_Recommender
from Recommenders.Hybrids.BaseHybridMultiRecommender import BaseHybridMultiRecommender
from Recommenders.MatrixFactorization.IALSRecommender_implicit import IALSRecommender_implicit
from Recommenders.Recommender_utils import check_matrix
from Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender

output_root_path = "./result_experiments/"

class Hybrid_SLIM_EASE_R_IALS(BaseHybridMultiRecommender):
    """ Hybrid_SLIM_EASE_R_IALS
    Hybrid of N prediction scores R
    """

    RECOMMENDER_NAME = "Hybrid_SLIM_EASE_R_IALS"

    def __init__(self, URM_train):

        recommender_1 = SLIMElasticNetRecommender(URM_train)
        recommender_1.load_model(output_root_path, file_name="slimelastic_urmall_453.zip")

        recommender_2 = EASE_R_Recommender(URM_train)
        recommender_2.fit()

        recommender_3 = IALSRecommender_implicit(URM_train)
        recommender_3.fit()

        recommender_array = [recommender_1, recommender_2, recommender_3]


        super(Hybrid_SLIM_EASE_R_IALS, self).__init__(URM_train, recommender_array, 3)

    @ignore_warnings()
    def fit(self, alpha=0.8747407337453746, beta=0.3467502656911922, gamma=0.012212488901387934):
        self.weight_array = [alpha, beta, gamma]
