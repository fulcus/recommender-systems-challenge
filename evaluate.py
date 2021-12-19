import os
import traceback

import numpy as np
import scipy.sparse as sps

from Data_manager.split_functions.split_train_validation_random_holdout import \
    split_train_in_two_percentage_global_sample
from Evaluation.Evaluator import EvaluatorHoldout
from Recommenders.Hybrids.HybridRatings_SLIM_Rp3 import HybridRatings_SLIM_Rp3
from Recommenders.Hybrids.HybridSimilarity_SLIM_Rp3 import HybridSimilarity_SLIM_Rp3
from Recommenders.Hybrids.Hybrid_SlimElastic_Rp3 import Hybrid_SlimElastic_Rp3
from Recommenders.Hybrids.Hybrid_SlimElastic_Rp3_PureSVD import Hybrid_SlimElastic_Rp3_PureSVD
from Recommenders.Hybrids.others.ScoresHybridRP3betaKNNCBF import ScoresHybridRP3betaKNNCBF
from Recommenders.Hybrids.BaseHybridRatings import BaseHybridRatings
from Recommenders.Incremental_Training_Early_Stopping import Incremental_Training_Early_Stopping
from Recommenders.KNN.ItemKNNCBFWeightedSimilarityRecommender import ItemKNNCBFWeightedSimilarityRecommender
from Recommenders.MatrixFactorization.PureSVDRecommender import PureSVDItemRecommender
from Recommenders.KNN.ItemKNNCustomSimilarityRecommender import ItemKNNCustomSimilarityRecommender
from Recommenders.MatrixFactorization.IALSRecommender_implicit import IALSRecommender_implicit
from Recommenders.Recommender_import_list import *
from Recommenders.Recommender_utils import check_matrix
from reader import load_urm, load_icm, load_target
from run_all_algorithms import _get_instance
from sklearn import feature_extraction

res_dir = 'result_experiments/csv'
output_root_path = "./result_experiments/"

recommender_class_list = [
    # UserKNNCBFRecommender, # UCM needed
    # ItemKNNCBFRecommender,
    # ItemKNNCBFWeightedSimilarityRecommender,  # new
    # UserKNN_CFCBF_Hybrid_Recommender, # UCM needed
    # ItemKNN_CFCBF_Hybrid_Recommender,
    # SLIMElasticNetRecommender,  # too slow to train
    # UserKNNCFRecommender,
    # IALSRecommender,
    # MatrixFactorization_BPR_Cython,
    # MatrixFactorization_FunkSVD_Cython, # fix low values
    # MatrixFactorization_AsySVD_Cython, # fix low values
    # EASE_R_Recommender, # fix low values
    # ItemKNNCFRecommender,
    # P3alphaRecommender,
    # SLIM_BPR_Cython,
    # RP3betaRecommender,
    # PureSVDRecommender,
    # PureSVDItemRecommender
    # NMFRecommender,

    # LightFMCFRecommender,
    # LightFMUserHybridRecommender, # UCM needed
    # LightFMItemHybridRecommender,

    # Hybrid_SlimElastic_Rp3,
    # Hybrid_SlimElastic_Rp3_PureSVD,
    # Hybrid_SlimElastic_Rp3_ItemKNNCF

    # IALSRecommender_implicit
    # HybridWsparseSLIMRp3
    HybridRatings_SLIM_Rp3,
    HybridSimilarity_SLIM_Rp3
]

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

def evaluate_all_recommenders(URM_all, *ICMs):
    ICM_genre, ICM_subgenre, ICM_channel, ICM_event, ICM_all = ICMs
    # ICM_merged = load_merged_icm("ICM_merged.csv")

    # URM_train, URM_test = train_test_holdout(URM_all, train_perc=0.85)

    URM_train, URM_test = split_train_in_two_percentage_global_sample(URM_all=URM_all, train_percentage=0.90)

    # tmp = check_matrix(ICM_channel.T, 'csr', dtype=np.float32)
    # tmp = tmp.multiply(14)
    # URM_train = sps.vstack((URM_train, tmp), format='csr', dtype=np.float32)

    evaluator = EvaluatorHoldout(URM_test, cutoff_list=[10], exclude_seen=True)

    earlystopping_keywargs = {"validation_every_n": 2,
                              "stop_on_validation": True,
                              "evaluator_object": evaluator,
                              "lower_validations_allowed": 3,
                              "validation_metric": "MAP",
                              }

    for recommender_class in recommender_class_list:

        try:
            print("Algorithm: {}".format(recommender_class.RECOMMENDER_NAME))
            # URM_tfidf = feature_extraction.text.TfidfTransformer().fit_transform(URM_train)
            recommender_object = _get_instance(recommender_class, URM_train, ICM_channel)

            if isinstance(recommender_object, ItemKNNCBFWeightedSimilarityRecommender):
                fit_params = {"ICMs": ICMs}
            elif isinstance(recommender_object, ItemKNNCFRecommender):
                fit_params = {"topK": 200, "shrink": 200, "feature_weighting": "TF-IDF"}
            elif isinstance(recommender_object, SLIMElasticNetRecommender):
                fit_params = {"topK": 453, 'l1_ratio': 0.00029920499017254754, 'alpha': 0.10734084960757517}
            elif isinstance(recommender_object, IALSRecommender_implicit):
                # 'num_factors': 11, 'epochs': 36, 'confidence_scaling': 'linear',
                # 'alpha': 0.0798650671543897, 'epsilon': 0.00205004763058707, 'reg': 0.008433763108035943
                fit_params = {'n_factors' : 768, 'regularization' : 0.4489004525533907, 'iterations':76 }
            elif isinstance(recommender_object, RP3betaRecommender):
                fit_params = {'topK': 40, 'alpha': 0.4208737801266599, 'beta': 0.5251543657397256,
                              'normalize_similarity': True}
            elif isinstance(recommender_object, MultVAERecommender):
                fit_params = {'topK': 615, 'l1_ratio': 0.007030044688343361, 'alpha': 0.07010526286528686}
            elif isinstance(recommender_object, Hybrid_SlimElastic_Rp3):
                fit_params = {'alpha': 0.9}
            elif isinstance(recommender_object, HybridRatings_SLIM_Rp3):
                fit_params = {'alpha': 0.9}
            elif isinstance(recommender_object, HybridSimilarity_SLIM_Rp3):
                fit_params = {'alpha': 0.9610229519605884, 'topK': 1199}
            elif isinstance(recommender_object, PureSVDRecommender):
                fit_params = {'num_factors' :29}
            elif isinstance(recommender_object, Hybrid_SlimElastic_Rp3_PureSVD):
                fit_params = {'alpha': 0.9, 'beta': 0.1, 'gamma': 0.1}
            else:
                fit_params = {}

            recommender_object.fit(**fit_params)
            results_run_1, results_run_string_1 = evaluator.evaluateRecommender(recommender_object)
            #recommender_object.save_model(output_root_path, file_name="temp_model.zip")
            # recommender_object = _get_instance(recommender_class, URM_train, ICM_all)
            # recommender_object.load_model(output_root_path, file_name="temp_model.zip")
            # os.remove(output_root_path + "temp_model.zip")
            # results_run_2, results_run_string_2 = evaluator.evaluateRecommender(recommender_object)

            print("1-Algorithm: {}, results: \n{}".format(recommender_class.RECOMMENDER_NAME, results_run_string_1))
            logFile.write(
                "1-Algorithm: {}, results: \n{}\n".format(recommender_class.RECOMMENDER_NAME, results_run_string_1))

            # print("2-Algorithm: {}, results: \n{}".format(recommender_class.RECOMMENDER_NAME, results_run_string_2))
            # logFile.write(
            #     "2-Algorithm: {}, results: \n{}\n".format(recommender_class.RECOMMENDER_NAME, results_run_string_2))
            # if recommender_class not in [Random]:
            #     assert results_run_1.equals(results_run_2)
            logFile.flush()


        except Exception as e:
            traceback.print_exc()
            logFile.write("Algorithm: {} - Exception: {}\n".format(recommender_class.RECOMMENDER_NAME, str(e)))
            logFile.flush()


def evaluate_best_saved_model(URM_all):
    URM_train, URM_test = split_train_in_two_percentage_global_sample(URM_all=URM_all, train_percentage=0.90)

    evaluator = EvaluatorHoldout(URM_test, cutoff_list=[10], exclude_seen=True)

    # set here the recommender you want to use
    recommender_object = SLIMElasticNetRecommender(URM_train) # SLIMElasticNetRecommender(URM_train)

    # rec_best_model_last.zip is the output of the run_hyperparameter_search (one best model for each rec class)
    # recommender_object.load_model(output_root_path, file_name=recommender_object.RECOMMENDER_NAME + "_best_model.zip")
    recommender_object.load_model(output_root_path, file_name="slimelastic_urmall.zip")

    results_run_1, results_run_string_1 = evaluator.evaluateRecommender(recommender_object)

    print("1-Algorithm: {}, results: \n{}".format(recommender_object.RECOMMENDER_NAME, results_run_string_1))
    logFile.write(
        "1-Algorithm: {}, results: \n{}\n".format(recommender_object.RECOMMENDER_NAME, results_run_string_1))


if __name__ == '__main__':
    URM_all, user_id_unique, item_id_unique = load_urm()

    ICM_genre = load_icm("data_ICM_genre.csv", weight=1)
    ICM_subgenre = load_icm("data_ICM_subgenre.csv", weight=1)
    ICM_channel = load_icm("data_ICM_channel.csv", weight=1)
    ICM_event = load_icm("data_ICM_event.csv", weight=1)

    ICM_all = sps.hstack([ICM_genre, ICM_subgenre, ICM_channel, ICM_event]).tocsr()
    ICMs = [ICM_genre, ICM_subgenre, ICM_channel, ICM_event, ICM_all]

    target_ids = load_target()

    # evaluate_best_saved_model(URM_all)
    evaluate_all_recommenders(URM_all, *ICMs)
