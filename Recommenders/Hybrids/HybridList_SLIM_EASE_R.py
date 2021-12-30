import numpy as np
from scipy import sparse as sps

from Recommenders.EASE_R.EASE_R_Recommender import EASE_R_Recommender
from Recommenders.Hybrids.BaseHybridList import BaseHybridList
from Recommenders.Hybrids.BaseHybridRatings import BaseHybridRatings
from Recommenders.MatrixFactorization.PureSVDRecommender import PureSVDRecommender
from Recommenders.Recommender_utils import check_matrix
from Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender
from reader import load_icm

output_root_path = "./result_experiments/"


class HybridList_SLIM_EASE_R(BaseHybridList):
    RECOMMENDER_NAME = "HybridList_SLIM_EASE_R"

    def __init__(self, URM_train):
        self.recommender_1 = SLIMElasticNetRecommender(URM_train)
        self.recommender_1.load_model(output_root_path, file_name="slimelastic_urmall_453.zip")

        ICM = load_icm("data_ICM_event.csv", weight=1)
        ICM = check_matrix(ICM.T, 'csr', dtype=np.float32)
        URM_ICM = sps.vstack((URM_train, ICM), format='csr', dtype=np.float32)

        # self.recommender_2 = PureSVDRecommender(URM_train)
        self.recommender_2 = EASE_R_Recommender(URM_ICM)
        self.recommender_2.fit()

        super(HybridList_SLIM_EASE_R, self).__init__(URM_train, self.recommender_1, self.recommender_2)
