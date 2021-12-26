import numpy as np

from Recommenders.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender
from Recommenders.EASE_R.EASE_R_Recommender import EASE_R_Recommender
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from Recommenders.Hybrids.BaseHybridRatings import BaseHybridRatings
from Recommenders.Hybrids.HybridRatings_PureSVD_EASE_R import HybridRatings_PureSVD_EASE_R
from Recommenders.Hybrids.HybridSimilarity_SLIM_Rp3 import HybridSimilarity_SLIM_Rp3
from Recommenders.MatrixFactorization.PureSVDRecommender import PureSVDRecommender
from Recommenders.Recommender_utils import check_matrix
from Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender

output_root_path = "./result_experiments/"


class HybridRatings_SLIM_PureSVD_EASE_R(BaseHybridRatings):

    RECOMMENDER_NAME = "HybridRatings_SLIM_PureSVD_EASE_R"

    def __init__(self, URM_train):

        self.recommender_1 = SLIMElasticNetRecommender(URM_train)
        self.recommender_1.load_model(output_root_path, file_name="slimelastic_urmall_453.zip")

        self.recommender_2 = HybridRatings_PureSVD_EASE_R(URM_train)
        self.recommender_2.fit(alpha=0.5)

        super(HybridRatings_SLIM_PureSVD_EASE_R, self).__init__(URM_train, self.recommender_1, self.recommender_2)
