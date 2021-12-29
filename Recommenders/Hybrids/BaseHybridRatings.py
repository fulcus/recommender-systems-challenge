import numpy as np

from Recommenders.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from Recommenders.Recommender_utils import check_matrix
from Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender

output_root_path = "./result_experiments/"


def _normalize(item_weights_1, item_weights_2):
    mean1 = np.mean(item_weights_1)
    mean2 = np.mean(item_weights_2)
    std1 = np.std(item_weights_1)
    std2 = np.std(item_weights_2)
    if std1 != 0 and std2 != 0:
        item_weights_1 = (item_weights_1 - mean1) / std1
        item_weights_2 = (item_weights_2 - mean2) / std2

    return item_weights_1, item_weights_2


class BaseHybridRatings(BaseItemSimilarityMatrixRecommender):
    """
    Hybrid of two prediction scores R = R1*alpha + R2*(1-alpha)
    """

    RECOMMENDER_NAME = "BaseHybridRatings"

    def __init__(self, URM_train, recommender_1, recommender_2):
        super(BaseHybridRatings, self).__init__(URM_train)
        self.beta = None
        self.alpha = None
        self.URM_train = check_matrix(URM_train.copy(), 'csr')

        self.recommender_1 = recommender_1
        self.recommender_2 = recommender_2

    def fit(self, alpha=0.95, beta=0.05, alpha1=0.96, beta1= 0.04, topK1=1199):
        self.alpha = alpha
        self.beta = beta

        self.recommender_1.fit(alpha=alpha1, beta=beta1, topK=topK1)
        self.recommender_2.fit()

        print('{} hyperparam: alpha: {}'.format(self.RECOMMENDER_NAME, alpha))

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        item_weights_1 = self.recommender_1._compute_item_score(user_id_array)
        item_weights_2 = self.recommender_2._compute_item_score(user_id_array)

        item_weights_1, item_weights_2 = _normalize(item_weights_1, item_weights_2)

        item_weights = item_weights_1 * self.alpha + item_weights_2 * self.beta

        return item_weights
