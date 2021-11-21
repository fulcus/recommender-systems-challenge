import traceback

import numpy as np
from datetime import datetime
import os
import scipy.sparse as sps
from Data_manager.DataSplitter import DataSplitter
from Data_manager.TVShows.TVShowsReader import TVShowsReader
from Evaluation.Evaluator import Evaluator, EvaluatorHoldout
from Recommenders.BaseRecommender import BaseRecommender
from Recommenders.Incremental_Training_Early_Stopping import Incremental_Training_Early_Stopping
from Recommenders.Recommender_import_list import *
from reader import load_urm, load_icm_asset, load_target
from run_all_algorithms import _get_instance

res_dir = 'Results'

recommender_class_list = [
    Random,
    TopPop,
    GlobalEffects,
    SLIMElasticNetRecommender,
    UserKNNCFRecommender,
    IALSRecommender,
    MatrixFactorization_BPR_Cython,
    MatrixFactorization_FunkSVD_Cython,
    MatrixFactorization_AsySVD_Cython,
    EASE_R_Recommender,
    ItemKNNCFRecommender,
    P3alphaRecommender,
    SLIM_BPR_Cython,
    RP3betaRecommender,
    PureSVDRecommender,
    NMFRecommender,
    UserKNNCBFRecommender,
    ItemKNNCBFRecommender,
    UserKNN_CFCBF_Hybrid_Recommender,
    ItemKNN_CFCBF_Hybrid_Recommender,
    LightFMCFRecommender,
    LightFMUserHybridRecommender,
    LightFMItemHybridRecommender,
]



output_root_path = "./result_experiments/"

# If directory does not exist, create
if not os.path.exists(output_root_path):
    os.makedirs(output_root_path)

logFile = open(output_root_path + "result_all_algorithms.txt", "a")


def train_test_holdout(URM_all, train_perc=0.8):
    numInteractions = URM_all.nnz
    URM_all = URM_all.tocoo()
    shape = URM_all.shape

    train_mask = np.random.choice([True, False], numInteractions, p=[train_perc, 1 - train_perc])

    URM_train = sps.coo_matrix((URM_all.data[train_mask], (URM_all.row[train_mask], URM_all.col[train_mask])),
                               shape=shape)
    URM_train = URM_train.tocsr()

    test_mask = np.logical_not(train_mask)

    URM_test = sps.coo_matrix((URM_all.data[test_mask], (URM_all.row[test_mask], URM_all.col[test_mask])), shape=shape)
    URM_test = URM_test.tocsr()

    return URM_train, URM_test


def create_csv(target_ids, results, rec_name):
    exp_dir = os.path.join(res_dir, rec_name)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    csv_fname = 'results_'
    csv_fname += datetime.now().strftime('%b%d_%H-%M-%S') + '.csv'

    with open(os.path.join(exp_dir, csv_fname), 'w') as f:
        f.write('user_id,item_list\n')
        for target_id, result in zip(target_ids, results):
            f.write(str(target_id) + ', ' + ' '.join(map(str, result)) + '\n')


def run_all_data_train():
    URM_all, user_id_unique, item_id_unique = load_urm()
    ICM_all = load_icm_asset()
    target_ids = load_target()

    URM_train, URM_test = train_test_holdout(URM_all, train_perc=0.9)

    evaluator = EvaluatorHoldout(URM_test, [10], exclude_seen=True)

    earlystopping_keywargs = {"validation_every_n": 2,
                              "stop_on_validation": True,
                              "evaluator_object": evaluator,
                              "lower_validations_allowed": 3,
                              "validation_metric": "MAP",
                              }

    for recommender_class in recommender_class_list:

        try:
            print("Algorithm: {}".format(recommender_class))
            recommender_object = _get_instance(recommender_class, URM_train, ICM_all)

            if isinstance(recommender_object, Incremental_Training_Early_Stopping):
                fit_params = {"epochs": 15, **earlystopping_keywargs}
            else:
                fit_params = {}

            recommender_object.fit(**fit_params)

            results_run_1, results_run_string_1 = evaluator.evaluateRecommender(recommender_object)

            recommender_object.save_model(output_root_path, file_name="temp_model.zip")

            recommender_object = _get_instance(recommender_class, URM_train, ICM_all)
            recommender_object.load_model(output_root_path, file_name="temp_model.zip")

            os.remove(output_root_path + "temp_model.zip")

            results_run_2, results_run_string_2 = evaluator.evaluateRecommender(recommender_object)

            # added for prediction
            item_list = recommender_object.recommend(target_ids, cutoff=10)
            create_csv(target_ids, item_list, str(recommender_class))

            if recommender_class not in [Random]:
                assert results_run_1.equals(results_run_2)

            print("Algorithm: {}, results: \n{}".format(recommender_class, results_run_string_1))
            logFile.write("Algorithm: {}, results: \n{}\n".format(recommender_class, results_run_string_1))
            logFile.flush()


        except Exception as e:
            traceback.print_exc()
            logFile.write("Algorithm: {} - Exception: {}\n".format(recommender_class, str(e)))
            logFile.flush()




if __name__ == '__main__':
    run_all_data_train()
