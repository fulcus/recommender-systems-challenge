import numpy as np

from Recommenders.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender
from Recommenders.EASE_R.EASE_R_Recommender import EASE_R_Recommender
from Recommenders.Recommender_utils import check_matrix

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

    def __init__(self, URM_train, recommender_1, recommender_2, recommender_3, fold=None):
        super(BaseHybridRatings, self).__init__(URM_train)
        self.alpha = None
        self.beta = None
        self.gamma = None
        self.fold = fold

        self.URM_train = check_matrix(URM_train.copy(), 'csr')
        self.recommender_1 = recommender_1
        self.recommender_2 = recommender_2
        self.recommender_3 = recommender_3

    def fit(self, alpha=0.9560759641998946, beta=0.3, gamma=0.3, alpha1=0.9739242060693925, beta1=0.32744235125291515,
            topK1=837):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.recommender_1.fit(alpha=alpha1, beta=beta1, topK=topK1)

        if self.fold is not None and isinstance(self.recommender_2, EASE_R_Recommender):
            easer_name = 'EASE_R_Recommender-fold{}.zip'.format(self.fold)
            self.recommender_2.load_model(output_root_path, file_name=easer_name)
        else:
            self.recommender_2.fit()

        self.recommender_3.fit()

        print('{} hyperparams: {}, {}, {}'.format(self.RECOMMENDER_NAME, self.alpha, self.beta, self.gamma))

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        item_weights_1 = self.recommender_1._compute_item_score(user_id_array)
        item_weights_2 = self.recommender_2._compute_item_score(user_id_array)
        item_weights_3 = self.recommender_3._compute_item_score(user_id_array)

        item_weights_1, item_weights_2 = _normalize(item_weights_1, item_weights_2)
        item_weights_1, item_weights_3 = _normalize(item_weights_1, item_weights_3)
        item_weights_2, item_weights_3 = _normalize(item_weights_2, item_weights_3)

        item_weights = item_weights_1 * self.alpha + item_weights_2 * self.beta + item_weights_3 * self.gamma

        return item_weights
