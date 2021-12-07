import numpy as np
import operator
import os
import os
import traceback
import scipy.sparse as sps

from Evaluation.Evaluator import EvaluatorHoldout
from Recommenders.BaseRecommender import BaseRecommender
from Recommenders.DataIO import DataIO
from Recommenders.Incremental_Training_Early_Stopping import Incremental_Training_Early_Stopping
from Recommenders.KNN.ItemKNNCBFWeightedSimilarityRecommender import ItemKNNCBFWeightedSimilarityRecommender
from Recommenders.KNN.ItemKNNCustomSimilarityRecommender import ItemKNNCustomSimilarityRecommender
from Recommenders.Recommender_import_list import *
from Recommenders.Recommender_utils import check_matrix
from reader import load_urm, load_icm, load_target
from run_all_algorithms import _get_instance
from sklearn import feature_extraction

output_root_path = "./result_experiments/"


class Hybrid_SlimElastic_Rp3_ItemKNNCF(BaseRecommender):
    """Hybrid_SlimElastic_Rp3_ItemKNNCF"""

    RECOMMENDER_NAME = "HybridSlimElasticRp3ItemKNNCF"

    def __init__(self, URM_train):
        super(Hybrid_SlimElastic_Rp3_ItemKNNCF, self).__init__(URM_train)

        self.URM_train = check_matrix(URM_train.copy(), 'csr')
        self.slim = SLIMElasticNetRecommender(
            URM_train)
        self.rp3 = RP3betaRecommender(URM_train)
        self.itemcfknn = ItemKNNCFRecommender(URM_train)

    def fit(self, alpha=0.9, beta=0.1, gamma=0.1):
        self.slim.load_model(output_root_path, file_name="slimelastic_urmall.zip")
        # self.slim.load_model(output_root_path, file_name="saved_slim.zip")
        self.itemcfknn.fit(topK=200, shrink=200, feature_weighting="TF-IDF")
        # self.slim.fit(topK=453, l1_ratio=0.00029920499017254754, alpha=0.10734084960757517)
        self.rp3.fit(topK=40, alpha=0.4208737801266599, beta=0.5251543657397256, normalize_similarity=True)
        self.fit_no_cached(path_slim=None, path_rp3=None, alpha=alpha, beta=beta, gamma=gamma)

    def fit_cached(self, path_slim, path_rp3, path_cb, alpha=0.9, beta=0.1, gamma=0.1):
        mat_scores_slim = np.load(path_slim + '.npy')
        mat_scores_rp3 = np.load(path_rp3 + '.npy')
        mat_scores_knncb = np.load(path_cb + '.npy')

        self.score_matrix = alpha * mat_scores_slim + beta * mat_scores_rp3 + gamma * mat_scores_knncb

    def fit_no_cached(self, path_slim, path_rp3, alpha=0.9, beta=0.1, gamma=0.1):

        n_users = 13650
        n_item = 18059
        list_slim = []
        list_rp3 = []
        list_knncb = []
        for u_id in range(n_users):
            # if u_id % 1000 == 0:
            #     print('user: {} / {}'.format(u_id, n_users - 1))
            list_slim.append(np.squeeze(self.slim._compute_item_score(user_id_array=u_id)))
            list_rp3.append(np.squeeze(self.rp3._compute_item_score(user_id_array=u_id)))
            list_knncb.append(np.squeeze(self.itemcfknn._compute_item_score(user_id_array=u_id)))

        mat_scores_slim = np.stack(list_slim, axis=0)
        mat_scores_rp3 = np.stack(list_rp3, axis=0)
        mat_scores_knncb = np.stack(list_knncb, axis=0)

        print("slim scores stats:")
        print("min = {}".format(mat_scores_slim.min()))
        print("max = {}".format(mat_scores_slim.max()))
        print("average = {}".format(mat_scores_slim.mean()))
        print("rp3 scores stats:")
        print("min = {}".format(mat_scores_rp3.min()))
        print("max = {}".format(mat_scores_rp3.max()))
        print("average = {}".format(mat_scores_rp3.mean()))
        print("itemknncb scores stats:")
        print("min = {}".format(mat_scores_knncb.min()))
        print("max = {}".format(mat_scores_knncb.max()))
        print("average = {}".format(mat_scores_knncb.mean()))

        # normalization
        mat_scores_slim /= mat_scores_slim.max()
        mat_scores_rp3 /= mat_scores_rp3.max()
        mat_scores_knncb /= mat_scores_knncb.max()

        self.mat_scores_slim = mat_scores_slim
        self.mat_scores_rp3 = mat_scores_rp3
        self.mat_scores_knncb = mat_scores_knncb

        print("slim scores stats:")
        print("min = {}".format(mat_scores_slim.min()))
        print("max = {}".format(mat_scores_slim.max()))
        print("average = {}".format(mat_scores_slim.mean()))
        print("rp3 scores stats:")
        print("min = {}".format(mat_scores_rp3.min()))
        print("max = {}".format(mat_scores_rp3.max()))
        print("average = {}".format(mat_scores_rp3.mean()))
        print("itemknncb scores stats:")
        print("min = {}".format(mat_scores_knncb.min()))
        print("max = {}".format(mat_scores_knncb.max()))
        print("average = {}".format(mat_scores_knncb.mean()))

        # np.save(path_slim, arr=mat_scores_slim)
        # np.save(path_rp3, arr=mat_scores_rp3)

        self.score_matrix = alpha * self.mat_scores_slim + beta * self.mat_scores_rp3 + gamma * self.mat_scores_knncb

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        return self.score_matrix[user_id_array]

    def save_model(self, folder_path, file_name=None):

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        self._print("Saving model in file '{}'".format(folder_path + file_name))

        data_dict_to_save = {"score_matrix": self.score_matrix}

        dataIO = DataIO(folder_path=folder_path)
        dataIO.save_data(file_name=file_name, data_dict_to_save=data_dict_to_save)

        self._print("Saving complete")
