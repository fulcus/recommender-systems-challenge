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

    def recommend(self, user_id_array, cutoff=None, remove_seen_flag=True, items_to_compute=None,
                  remove_top_pop_flag=False, remove_custom_items_flag=False, return_scores=False):

        print('called recommend')

        # If is a scalar transform it in a 1-cell array
        if cutoff is None:
            cutoff = [10]
        if np.isscalar(user_id_array):
            user_id_array = np.atleast_1d(user_id_array)
            single_user = True
        else:
            single_user = False

        if cutoff is None:
            cutoff = self.URM_train.shape[1] - 1

        cutoff = min(cutoff, self.URM_train.shape[1] - 1)

        total_ranking_list = []

        # return scores or rec_1 to pass checks in evaluator, as they are not actually used
        _, return_scores_1 = self.recommender_1.recommend(user_id_array, cutoff=cutoff,
                                                          remove_seen_flag=remove_seen_flag,
                                                          items_to_compute=items_to_compute,
                                                          remove_top_pop_flag=remove_top_pop_flag,
                                                          remove_custom_items_flag=remove_custom_items_flag,
                                                          return_scores=True)
        # print('ranking_list_1: {}\nscores_batch_1: {}'.format(ranking_list_1, scores_batch_1))

        for user_id in user_id_array:
            ranking_list_1 = self.recommender_1.recommend(user_id, cutoff=cutoff,
                                                          remove_seen_flag=remove_seen_flag,
                                                          items_to_compute=items_to_compute,
                                                          remove_top_pop_flag=remove_top_pop_flag,
                                                          remove_custom_items_flag=remove_custom_items_flag,
                                                          return_scores=False)

            ranking_list_2 = self.recommender_2.recommend(user_id, cutoff=self.rec_2_evaluate_top_k,
                                                          remove_seen_flag=remove_seen_flag,
                                                          items_to_compute=items_to_compute,
                                                          remove_top_pop_flag=remove_top_pop_flag,
                                                          remove_custom_items_flag=remove_custom_items_flag,
                                                          return_scores=False)

            # print('ranking_list_1: {}\nranking_list_2: {}'.format(ranking_list_1, ranking_list_2))
            # print('scores_batch_1: {}\nscores_batch_2: {}'.format(scores_batch_1, scores_batch_2))

            user_ranking_list = np.array(ranking_list_1[:self.min_items_from_rec_1])

            # print('user list before: ' + str(user_ranking_list))

            list_diff = list(set(ranking_list_2) - set(ranking_list_1))
            # print('list difference: ' + str(list_diff))

            for item in list_diff:
                if user_ranking_list.shape[0] == cutoff:
                    break
                user_ranking_list = np.append(user_ranking_list, item)

            for item in ranking_list_1:
                if user_ranking_list.shape[0] == cutoff:
                    break
                if item not in user_ranking_list:
                    user_ranking_list = np.append(user_ranking_list, item)

            total_ranking_list.append(user_ranking_list)

            # print('final user ranking list: ' + str(total_ranking_list))

        # print('ranking_list_1: {}\nscores_batch_1: {}'.format(total_ranking_list, total_scores))

        if single_user:
            total_ranking_list = total_ranking_list[0]

        if return_scores:
            return total_ranking_list, return_scores_1  # returned scores of rec_1 only (wrong)
        else:
            return total_ranking_list
