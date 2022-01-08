from Recommenders.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender
from Recommenders.Recommender_utils import check_matrix


def merge(ranks1, ranks2):
    new_ranks = []
    for el1, el2 in zip(ranks1[0], ranks2[0]):
        new_el = []
        l = 0
        m = 0
        i = 0
        while i < 10:
            if el1[l] not in new_el:
                new_el.append(el1[l])
                i += 1
            l += 1
            if el2[m] not in new_el:
                new_el.append(el2[m])
                i += 1
            m += 1
        new_ranks.append(new_el)

    # Scores are not correct
    return [new_ranks, ranks1[1]]


class RankingHybrid(BaseItemSimilarityMatrixRecommender):
    """ ItemKNNScoresHybridRecommender
    Hybrid of two prediction scores R = R1*alpha + R2*(1-alpha)
    NB: Rec_1 is itemKNNCF, Rec_2 is userKNNCF

    """

    RECOMMENDER_NAME = "RankingHybrid"

    def __init__(self, URM_train, Recommender_1, Recommender_2):
        super(RankingHybrid, self).__init__(URM_train)

        self.URM_train = check_matrix(URM_train.copy(), 'csr')
        self.Recommender_1 = Recommender_1
        self.Recommender_2 = Recommender_2

    def recommend(self, user_id_array, cutoff=None, remove_seen_flag=True, items_to_compute=None,
                  remove_top_pop_flag=False, remove_custom_items_flag=False, return_scores=False):
        ranks1 = self.Recommender_1.recommend(user_id_array, cutoff=cutoff, remove_seen_flag=remove_seen_flag,
                                              items_to_compute=items_to_compute,
                                              remove_top_pop_flag=remove_top_pop_flag,
                                              remove_custom_items_flag=remove_custom_items_flag,
                                              return_scores=return_scores)
        ranks2 = self.Recommender_2.recommend(user_id_array, cutoff=cutoff, remove_seen_flag=remove_seen_flag,
                                              items_to_compute=items_to_compute,
                                              remove_top_pop_flag=remove_top_pop_flag,
                                              remove_custom_items_flag=remove_custom_items_flag,
                                              return_scores=return_scores)
        merged_ranks = merge(ranks1, ranks2)

        return merged_ranks
