#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from Recommenders.Hybrids.BaseHybridSimilarity import BaseHybridSimilarity
from Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender

output_root_path = "./result_experiments/"


class HybridSimilarity_SLIM_Rp3(BaseHybridSimilarity):
    RECOMMENDER_NAME = "HybridSimilarity_SLIM_Rp3"

    def __init__(self, URM_train, fold=None):
        slim = SLIMElasticNetRecommender(URM_train)
        if fold is not None:
            slim_name = 'SLIMElasticNetRecommender-fold{}.zip'.format(fold)
        else:
            slim_name = 'slim742.zip'
        slim.load_model(output_root_path, file_name=slim_name)
        # self.slim.fit(topK=453, l1_ratio=0.00029920499017254754, alpha=0.10734084960757517)

        rp3 = RP3betaRecommender(URM_train)
        rp3.fit(topK=40, alpha=0.4208737801266599, beta=0.5251543657397256, normalize_similarity=True)

        super(HybridSimilarity_SLIM_Rp3, self).__init__(URM_train, slim, rp3)
