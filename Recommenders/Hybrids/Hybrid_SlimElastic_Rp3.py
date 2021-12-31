import numpy as np

from Recommenders.BaseRecommender import BaseRecommender
from Recommenders.DataIO import DataIO
from Recommenders.Recommender_import_list import *
from Recommenders.Recommender_utils import check_matrix

output_root_path = "./result_experiments/"


class Hybrid_SlimElastic_Rp3(BaseRecommender):
    RECOMMENDER_NAME = "HybridSlimElasticRp3"

    def __init__(self, URM_train):
        super(Hybrid_SlimElastic_Rp3, self).__init__(URM_train)

        self.URM_train = check_matrix(URM_train.copy(), 'csr')
        self.slim = SLIMElasticNetRecommender(
            URM_train)
        self.rp3 = RP3betaRecommender(URM_train)

    def fit(self, alpha=0.5):
        self.slim.load_model(output_root_path, file_name="slim742.zip")

        # self.slim.fit(topK=453, l1_ratio=0.00029920499017254754, alpha=0.10734084960757517)
        self.rp3.fit(topK=40, alpha=0.4208737801266599, beta=0.5251543657397256, normalize_similarity=True)
        self.fit_no_cached(path_slim=None, path_rp3=None, alpha=alpha)

    def fit_cached(self, path_slim, path_rp3, alpha=0.5):
        mat_scores_slim = np.load(path_slim + '.npy')
        mat_scores_rp3 = np.load(path_rp3 + '.npy')

        self.score_matrix = alpha * mat_scores_slim + (1 - alpha) * mat_scores_rp3

    def fit_no_cached(self, path_slim, path_rp3, alpha=0.5):

        n_users = 13650
        n_item = 18059
        list_slim = []
        list_rp3 = []
        for u_id in range(n_users):
            # if u_id % 1000 == 0:
            #     print('user: {} / {}'.format(u_id, n_users - 1))
            list_slim.append(np.squeeze(self.slim._compute_item_score(user_id_array=u_id)))
            list_rp3.append(np.squeeze(self.rp3._compute_item_score(user_id_array=u_id)))

        mat_scores_slim = np.stack(list_slim, axis=0)
        mat_scores_rp3 = np.stack(list_rp3, axis=0)

        print("slim scores stats:")
        print("min = {}".format(mat_scores_slim.min()))
        print("max = {}".format(mat_scores_slim.max()))
        print("average = {}".format(mat_scores_slim.mean()))
        print("rp3 scores stats:")
        print("min = {}".format(mat_scores_rp3.min()))
        print("max = {}".format(mat_scores_rp3.max()))
        print("average = {}".format(mat_scores_rp3.mean()))

        # normalization
        mat_scores_slim /= mat_scores_slim.max()
        mat_scores_rp3 /= mat_scores_rp3.max()

        self.mat_scores_slim = mat_scores_slim
        self.mat_scores_rp3 = mat_scores_rp3

        print("slim scores stats:")
        print("min = {}".format(mat_scores_slim.min()))
        print("max = {}".format(mat_scores_slim.max()))
        print("average = {}".format(mat_scores_slim.mean()))
        print("rp3 scores stats:")
        print("min = {}".format(mat_scores_rp3.min()))
        print("max = {}".format(mat_scores_rp3.max()))
        print("average = {}".format(mat_scores_rp3.mean()))

        # np.save(path_slim, arr=mat_scores_slim)
        # np.save(path_rp3, arr=mat_scores_rp3)

        self.score_matrix = alpha * self.mat_scores_slim + (1 - alpha) * self.mat_scores_rp3

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
