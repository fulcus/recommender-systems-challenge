import numpy as np

from Recommenders.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from Recommenders.Hybrids.BaseHybridRatings import BaseHybridRatings
from Recommenders.Recommender_utils import check_matrix
from Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender

output_root_path = "./result_experiments/"


class HybridRatings_SLIM_Rp3(BaseHybridRatings):

    RECOMMENDER_NAME = "HybridRatings_SLIM_PureSVD"

    def __init__(self, URM_train):
        self.recommender_1 = SLIMElasticNetRecommender(URM_train)
        self.recommender_1.load_model(output_root_path, file_name="slim_splitforeval742.zip")

        self.recommender_2 = RP3betaRecommender(URM_train)
        self.recommender_2.fit(topK=40, alpha=0.4208737801266599, beta=0.5251543657397256, normalize_similarity=True)

        super(HybridRatings_SLIM_Rp3, self).__init__(URM_train, self.recommender_1, self.recommender_2)
