import numpy as np

from Recommenders.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender
from Recommenders.NonPersonalizedRecommender import TopPop
from Recommenders.Recommender_utils import check_matrix
from Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender

output_root_path = "./result_experiments/"


class HybridGrouping_SLIM_TopPop(BaseItemSimilarityMatrixRecommender):
    RECOMMENDER_NAME = "HybridGrouping_SLIM_TopPop"

    def __init__(self, URM_train):
        super(BaseItemSimilarityMatrixRecommender, self).__init__(URM_train)
        self.alpha = None
        self.URM_train = check_matrix(URM_train.copy(), 'csr')

        self.recommender_1 = SLIMElasticNetRecommender(URM_train)
        self.recommender_1.load_model(output_root_path, file_name="slimelastic_urmall_453.zip")

        self.recommender_2 = TopPop(URM_train)
        self.recommender_2.fit()

    def fit(self, alpha=0.95):
        self.alpha = alpha

        print('{} hyperparam: alpha: {}'.format(self.RECOMMENDER_NAME, alpha))

    def _compute_item_score(self, user_id_array, items_to_compute=None):

        item_weights = np.empty([len(user_id_array), self.URM_train.shape[1]])

        for i in range(len(user_id_array)):

            interactions = len(self.URM_train[user_id_array[i], :].indices)

            if interactions > 50:
                w1 = self.recommender_1._compute_item_score(user_id_array[i], items_to_compute)
                # w1 /= np.linalg.norm(w1, 2)
                # w2 = self.ItemCF2._compute_item_score(user_id_array[i], items_to_compute)
                # w2 /= np.linalg.norm(w2, 2)
                # w = w1 + w2
                item_weights[i, :] = w1
            else:
                w1 = self.recommender_2._compute_item_score([user_id_array[i]], items_to_compute)
                # w2 = self.ItemCF1._compute_item_score(user_id_array[i], items_to_compute)
                # w3 = w1 * 3.1354787809646 + w2 * 0.6847368170848224
                # w3 /= np.linalg.norm(w3, 2)
                # w4 = self.Als1._compute_item_score(user_id_array[i], items_to_compute)
                # w4 /= np.linalg.norm(w4, 2)
                # w = w3 + w4 * 1.75
                item_weights[i, :] = w1

        return item_weights
