import numpy as np

from Recommenders.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender
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


class BaseHybridList(BaseItemSimilarityMatrixRecommender):
    """
    Hybrid of two prediction scores R = R1*alpha + R2*(1-alpha)
    """

    RECOMMENDER_NAME = "BaseHybridList"

    def __init__(self, URM_train, recommender_1, recommender_2):
        super(BaseHybridList, self).__init__(URM_train)
        self.items_from_rec_1 = None
        self.URM_train = check_matrix(URM_train.copy(), 'csr')

        self.recommender_1 = recommender_1
        self.recommender_2 = recommender_2

    def fit(self, min_items_from_rec_1=8, rec_2_evaluate_top_k=10):
        """
        :param min_items_from_rec_1: minimum number of items chosen by recommender_1
        :param rec_2_evaluate_top_k: number of items by recommender 2 to evaluate, to insert in recommendation list if not already present
        """
        self.min_items_from_rec_1 = min_items_from_rec_1
        self.rec_2_evaluate_top_k = rec_2_evaluate_top_k

        print('{} hyperparam: min_items_from_rec_1: {}'.format(self.RECOMMENDER_NAME, min_items_from_rec_1))

    def recommend(self, user_id_array, cutoff=10, remove_seen_flag=True, items_to_compute=None,
                  remove_top_pop_flag=False, remove_custom_items_flag=False, return_scores=False):

        ranking_list_1 = self.recommender_1.recommend(user_id_array, user_id_array, cutoff=cutoff,
                                                      remove_seen_flag=remove_seen_flag,
                                                      items_to_compute=items_to_compute,
                                                      remove_top_pop_flag=remove_top_pop_flag,
                                                      remove_custom_items_flag=remove_custom_items_flag,
                                                      return_scores=return_scores)

        ranking_list_2 = self.recommender_2.recommend(user_id_array, user_id_array, cutoff=self.rec_2_evaluate_top_k,
                                                      remove_seen_flag=remove_seen_flag,
                                                      items_to_compute=items_to_compute,
                                                      remove_top_pop_flag=remove_top_pop_flag,
                                                      remove_custom_items_flag=remove_custom_items_flag,
                                                      return_scores=return_scores)

        print('ranking_list_1: {}\nranking_list_2: {}'.format(ranking_list_1, ranking_list_2))

        total_ranking_list = ranking_list_1[:self.min_items_from_rec_1]

        for item in ranking_list_2:
            if item not in total_ranking_list:
                total_ranking_list.append(item)

        while len(total_ranking_list) < cutoff:
            for item in ranking_list_1:
                if item not in total_ranking_list:
                    total_ranking_list.append(item)

        print('final ranking list: ' + total_ranking_list)

        return total_ranking_list
