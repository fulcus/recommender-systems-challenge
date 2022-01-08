import numpy as np
from scipy import sparse as sps

from Recommenders.EASE_R.EASE_R_Recommender import EASE_R_Recommender
from Recommenders.Hybrids.BaseHybridRatings import BaseHybridRatings
from Recommenders.Hybrids.HybridSimilarity_SLIM_Rp3 import HybridSimilarity_SLIM_Rp3
from Recommenders.MatrixFactorization.IALSRecommender_implicit import IALSRecommender_implicit
from Recommenders.Recommender_utils import check_matrix
from reader import load_icm


class HybridRatings_IALS_hybrid_EASE_R_hybrid_SLIM_Rp3(BaseHybridRatings):
    RECOMMENDER_NAME = "HybridRatings_IALS_hybrid_EASE_R_hybrid_SLIM_Rp3"

    def __init__(self, URM_train, fold=None):
        self.recommender_1 = HybridSimilarity_SLIM_Rp3(URM_train, fold)
        ICM_event = load_icm("data_ICM_event.csv", weight=1)
        ICM_event = check_matrix(ICM_event.T, 'csr', dtype=np.float32)
        URM_ICM_event = sps.vstack((URM_train, ICM_event), format='csr', dtype=np.float32)
        self.recommender_2 = EASE_R_Recommender(URM_ICM_event)
        self.recommender_3 = IALSRecommender_implicit(URM_train)

        super(HybridRatings_IALS_hybrid_EASE_R_hybrid_SLIM_Rp3, self) \
            .__init__(URM_train, self.recommender_1, self.recommender_2, self.recommender_3, fold)
