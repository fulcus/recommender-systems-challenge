import numpy as np
from scipy import sparse as sps

from Recommenders.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender
from Recommenders.EASE_R.EASE_R_Recommender import EASE_R_Recommender
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from Recommenders.Hybrids.BaseHybridRatings import BaseHybridRatings
from Recommenders.Hybrids.HybridRatings_EASE_R_hybrid_SLIM_Rp3 import HybridRatings_EASE_R_hybrid_SLIM_Rp3
from Recommenders.Hybrids.HybridSimilarity_SLIM_Rp3 import HybridSimilarity_SLIM_Rp3
from Recommenders.MatrixFactorization.IALSRecommender_implicit import IALSRecommender_implicit
from Recommenders.MatrixFactorization.PureSVDRecommender import PureSVDRecommender
from Recommenders.Recommender_utils import check_matrix
from Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender
from reader import load_icm

output_root_path = "./result_experiments/"


class HybridRatings_IALS_hybrid_EASE_R_hybrid_SLIM_Rp3(BaseHybridRatings):

    RECOMMENDER_NAME = "HybridRatings_IALS_hybrid_EASE_R_hybrid_SLIM_Rp3"

    def __init__(self, URM_train):

        self.recommender_1 = HybridSimilarity_SLIM_Rp3(URM_train)

        ICM = load_icm("data_ICM_event.csv", weight=1)
        tmp = check_matrix(ICM.T, 'csr', dtype=np.float32)
        URM_ICM = sps.vstack((URM_train, tmp), format='csr', dtype=np.float32)
        self.recommender_2 = EASE_R_Recommender(URM_ICM)

        self.recommender_3 = IALSRecommender_implicit(URM_train)


        super(HybridRatings_IALS_hybrid_EASE_R_hybrid_SLIM_Rp3, self).__init__(URM_train, self.recommender_1, self.recommender_2,self.recommender_3)