from Recommenders.Recommender_utils import check_matrix
from Recommenders.BaseCBFRecommender import BaseItemCBFRecommender
from Recommenders.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender
from Recommenders.IR_feature_weighting import okapi_BM_25, TF_IDF
import numpy as np

from Recommenders.Similarity.Compute_Similarity import Compute_Similarity


class ItemKNNCBFWeightedSimilarityRecommender(BaseItemCBFRecommender, BaseItemSimilarityMatrixRecommender):
    """ ItemKNN recommender"""

    RECOMMENDER_NAME = "ItemKNNCBFCustomizedSimilarityRecommender"

    FEATURE_WEIGHTING_VALUES = ["BM25", "TF-IDF", "none"]

    def __init__(self, URM_train, ICM_train, verbose=True):
        super(ItemKNNCBFWeightedSimilarityRecommender, self).__init__(URM_train, ICM_train, verbose=verbose)

    def fit(self, topK=50, shrink=100, similarity='cosine', normalize=True, feature_weighting="none", ICM_bias=None,
            ICMs=None, ICMs_weights=None,
            **similarity_args):

        if ICMs_weights is None:
            ICMs_weights = [0.25, 0.25, 0.25, 0.25]
        if ICMs is None:
            ICMs = []

        self.topK = topK
        self.shrink = shrink

        if feature_weighting not in self.FEATURE_WEIGHTING_VALUES:
            raise ValueError(
                "Value for 'feature_weighting' not recognized. Acceptable values are {}, provided was '{}'".format(
                    self.FEATURE_WEIGHTING_VALUES, feature_weighting))

        if ICM_bias is not None:
            self.ICM_train.data += ICM_bias

        if feature_weighting == "BM25":
            self.ICM_train = self.ICM_train.astype(np.float32)
            self.ICM_train = okapi_BM_25(self.ICM_train)

        elif feature_weighting == "TF-IDF":
            self.ICM_train = self.ICM_train.astype(np.float32)
            self.ICM_train = TF_IDF(self.ICM_train)

        for i in ICMs:
            i = check_matrix(i.copy(), 'csr', dtype=np.float32)
            i.eliminate_zeros()

        ICM_genre, ICM_subgenre, ICM_channel, ICM_event, _ = ICMs

        similarity_genre = Compute_Similarity(ICM_genre.T, shrink=shrink, topK=topK, normalize=normalize,
                                              similarity=similarity, **similarity_args)

        similarity_subgenre = Compute_Similarity(ICM_subgenre.T, shrink=shrink, topK=topK, normalize=normalize,
                                                 similarity=similarity, **similarity_args)

        similarity_channel = Compute_Similarity(ICM_channel.T, shrink=shrink, topK=topK, normalize=normalize,
                                                similarity=similarity, **similarity_args)

        similarity_event = Compute_Similarity(ICM_event.T, shrink=shrink, topK=topK, normalize=normalize,
                                                similarity=similarity, **similarity_args)

        sg = similarity_genre.compute_similarity()
        ssg = similarity_subgenre.compute_similarity()
        sc = similarity_channel.compute_similarity()
        se = similarity_event.compute_similarity()

        self.W_sparse = sg * ICMs_weights[0] + ssg * ICMs_weights[1] + sc * ICMs_weights[2] + se * ICMs_weights[3]
        self.W_sparse = check_matrix(self.W_sparse, format='csr')
