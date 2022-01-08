import os
import traceback

import numpy as np
import scipy.sparse as sps
from tqdm import tqdm

from Utils.split_train_validation_random_holdout import \
    split_train_in_two_percentage_global_sample
from Evaluation.Evaluator import EvaluatorHoldout
from Recommenders.BaseCBFRecommender import BaseItemCBFRecommender
from Recommenders.Hybrids.HybridRatings_EASE_R_hybrid_SLIM_Rp3 import HybridRatings_EASE_R_hybrid_SLIM_Rp3
from Recommenders.Hybrids.HybridRatings_IALS_hybrid_EASE_R_hybrid_SLIM_Rp3 import \
    HybridRatings_IALS_hybrid_EASE_R_hybrid_SLIM_Rp3
from Recommenders.Hybrids.HybridRatings_PureSVD_EASE_R import HybridRatings_PureSVD_EASE_R
from Recommenders.Hybrids.HybridRatings_SLIM_EASE_R_PureSVD import HybridRatings_SLIM_PureSVD_EASE_R
from Recommenders.Hybrids.HybridRatings_SLIM_Rp3 import HybridRatings_SLIM_Rp3
from Recommenders.Hybrids.HybridSimilarity_SLIM_Rp3 import HybridSimilarity_SLIM_Rp3
from Recommenders.Hybrids.HybridSimilarity_withGroupedUsers import HybridSimilarity_withGroupedusers
from Recommenders.Hybrids.Hybrid_SLIM_EASE_R_IALS import Hybrid_SLIM_EASE_R_IALS
from Recommenders.Hybrids.Hybrid_SlimElastic_Rp3 import Hybrid_SlimElastic_Rp3
from Recommenders.Hybrids.Hybrid_SlimElastic_Rp3_PureSVD import Hybrid_SlimElastic_Rp3_PureSVD
from Recommenders.Hybrids.scores.ScoresHybridRP3betaKNNCBF import ScoresHybridRP3betaKNNCBF
from Recommenders.MatrixFactorization.IALSRecommender_implicit import IALSRecommender_implicit
from Recommenders.Recommender_import_list import *
from Recommenders.Recommender_utils import check_matrix
from reader import load_urm, load_icm, get_k_folds_URM

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
    # IALSRecommender_implicit,
    # MatrixFactorization_BPR_Cython,
    # MatrixFactorization_FunkSVD_Cython, # fix low values
    # MatrixFactorization_AsySVD_Cython, # fix low values
    # EASE_R_Recommender,
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
    # Hybrid_SLIM_EASE_R_IALS

    # HybridSimilarity_SLIM_Rp3,
    # HybridSimilarity_withSlimPerGroup
    # HybridSimilarity_withGroupedusers
    # HybridGrouping_SLIM_TopPop

    # HybridRatings_SLIM_Rp3,
    # HybridRatings_SLIM_EASE_R,
    # HybridRatings_EASE_R_hybrid_SLIM_Rp3
    # HybridRatings_PureSVD_EASE_R
    # HybridRatings_SLIM_PureSVD_EASE_R
    # HybridRatings_IALS_hybrid_EASE_R_hybrid_SLIM_Rp3
    # MultiRecommender
    HybridRatings_IALS_hybrid_EASE_R_hybrid_SLIM_Rp3
    # MultiRecommender
]

# If directory does not exist, create
if not os.path.exists(output_root_path):
    os.makedirs(output_root_path)

logFile = open(output_root_path + "result_all_algorithms.txt", "a")


def _get_instance(recommender_class, URM_train, ICM=None):
    if issubclass(recommender_class, BaseItemCBFRecommender) or issubclass(recommender_class,
                                                                           ScoresHybridRP3betaKNNCBF):
        recommender_object = recommender_class(URM_train, ICM)
    else:
        recommender_object = recommender_class(URM_train)

    return recommender_object


def _get_params(recommender_object):
    if isinstance(recommender_object, ItemKNNCFRecommender):
        fit_params = {'topK': 189, 'shrink': 0, 'similarity': 'cosine', 'normalize': True,
                      'feature_weighting': 'TF-IDF'}
    elif isinstance(recommender_object, ItemKNNCBFRecommender):
        fit_params = {'topK': 1000, 'shrink': 1000, 'similarity': 'cosine', 'normalize': False,
                      'feature_weighting': 'TF-IDF'}
    elif isinstance(recommender_object, SLIMElasticNetRecommender):
        fit_params = {"topK": 453, 'l1_ratio': 0.00029920499017254754, 'alpha': 0.10734084960757517}
    elif isinstance(recommender_object, IALSRecommender_implicit):
        fit_params = {'n_factors': 50, 'regularization': 0.001847510119137634}
    elif isinstance(recommender_object, RP3betaRecommender):
        fit_params = {'topK': 40, 'alpha': 0.4208737801266599, 'beta': 0.5251543657397256}
    elif isinstance(recommender_object, MultVAERecommender):
        fit_params = {'topK': 615, 'l1_ratio': 0.007030044688343361, 'alpha': 0.07010526286528686}
    elif isinstance(recommender_object, Hybrid_SlimElastic_Rp3):
        fit_params = {'alpha': 0.9}
    elif isinstance(recommender_object, HybridRatings_SLIM_Rp3):
        fit_params = {'alpha': 0.9}
    elif isinstance(recommender_object, HybridSimilarity_SLIM_Rp3):
        fit_params = {'alpha': 0.9610229519605884, 'topK': 1199}
    elif isinstance(recommender_object, HybridSimilarity_withGroupedusers):
        fit_params = {'alpha': 0.979326712891909, 'topK': 1349}
    elif isinstance(recommender_object, PureSVDRecommender):
        fit_params = {'num_factors': 28, 'random_seed': 0}
    elif isinstance(recommender_object, Hybrid_SlimElastic_Rp3_PureSVD):
        fit_params = {'alpha': 0.95, 'beta': 0.1, 'gamma': 0.1}
    elif isinstance(recommender_object, HybridRatings_EASE_R_hybrid_SLIM_Rp3):
        fit_params = {'alpha': 0.9610229519605884}
    elif isinstance(recommender_object, HybridRatings_PureSVD_EASE_R):
        fit_params = {'alpha': 0.5}
    elif isinstance(recommender_object, HybridRatings_SLIM_PureSVD_EASE_R):
        fit_params = {'alpha': 0.95}
    elif isinstance(recommender_object, Hybrid_SLIM_EASE_R_IALS):
        fit_params = {'alpha': 0.3815016492157693, 'beta': 0.5802064204762605, 'gamma': 0.06145838241599496}
    elif isinstance(recommender_object, HybridRatings_EASE_R_hybrid_SLIM_Rp3):
        fit_params = {'alpha': 0.95}
    elif isinstance(recommender_object, HybridRatings_IALS_hybrid_EASE_R_hybrid_SLIM_Rp3):
        # Average 5-fold MAP: 0.2480472, diff: 0.0020975
        fit_params = {'alpha': 0.9560759641998946, 'beta': 0.7550858561550403, 'gamma': 0.5227204586158875,
                      'alpha1': 0.9739242060693925, 'beta1': 0.32744235125291515, 'topK1': 837}
    else:
        fit_params = {}

    return fit_params


def evaluate_all_recommenders(URM_all, ICM=None):
    URM_train, URM_test = split_train_in_two_percentage_global_sample(URM_all=URM_all, train_percentage=0.8)
    evaluator = EvaluatorHoldout(URM_test, cutoff_list=[10])

    if ICM is not None:
        tmp = check_matrix(ICM.T, 'csr', dtype=np.float32)
        URM_train = sps.vstack((URM_train, tmp), format='csr', dtype=np.float32)

    for recommender_class in recommender_class_list:

        try:
            print("Algorithm: {}".format(recommender_class.RECOMMENDER_NAME))

            recommender_object = _get_instance(recommender_class, URM_train, ICM)
            fit_params = _get_params(recommender_object)
            recommender_object.fit(**fit_params)

            results_run, results_run_string = evaluator.evaluateRecommender(recommender_object)

            print("1-Algorithm: {}, results: \n{}".format(recommender_class.RECOMMENDER_NAME, results_run_string))
            logFile.write("1-Algorithm: {}, results: \n{}\n"
                          .format(recommender_class.RECOMMENDER_NAME, results_run_string))
            logFile.flush()

        except Exception as e:
            traceback.print_exc()
            logFile.write("Algorithm: {} - Exception: {}\n".format(recommender_class.RECOMMENDER_NAME, str(e)))
            logFile.flush()


def evaluate_best_saved_model(URM_all, ICM=None):
    URM_train, URM_test = split_train_in_two_percentage_global_sample(URM_all=URM_all, train_percentage=0.8)
    evaluator = EvaluatorHoldout(URM_test, cutoff_list=[10])

    if ICM is not None:
        tmp = check_matrix(ICM.T, 'csr', dtype=np.float32)
        URM_train = sps.vstack((URM_train, tmp), format='csr', dtype=np.float32)

    # set here the recommender you want to use
    recommender_object = SLIMElasticNetRecommender(URM_train)  # SLIMElasticNetRecommender(URM_train)

    # rec_best_model_last.zip is the output of the run_hyperparameter_search (one best model for each rec class)
    recommender_object.load_model(output_root_path, file_name="slim750group1.zip")

    results_run, results_run_string = evaluator.evaluateRecommender(recommender_object)

    print("Algorithm: {}, results: \n{}".format(recommender_object.RECOMMENDER_NAME, results_run_string))
    logFile.write("Algorithm: {}, results: \n{}\n".format(recommender_object.RECOMMENDER_NAME, results_run_string))
    logFile.flush()


def evaluate_kfold(k=5):
    URM_train_list, URM_test_list = get_k_folds_URM(k=k)
    evaluator_list = [EvaluatorHoldout(URM_test, cutoff_list=[10]) for URM_test in URM_test_list]

    MAP_list = []

    for i in tqdm(range(k), desc='Evaluating k folds'):
        recommender_object = HybridRatings_IALS_hybrid_EASE_R_hybrid_SLIM_Rp3(URM_train_list[i], fold=i)
        fit_params = _get_params(recommender_object)
        recommender_object.fit(**fit_params)

        results_run, results_run_string = evaluator_list[i].evaluateRecommender(recommender_object)
        MAP_list.append(float(results_run['MAP']))

        print("Fold {} - Algorithm: {}, results: \n{}"
              .format(i, recommender_object.RECOMMENDER_NAME, results_run_string))
        logFile.write("Fold {} -Algorithm: {}, results: \n{}\n"
                      .format(i, recommender_object.RECOMMENDER_NAME, results_run_string))

        # saved_name = "{}-fold{}".format(recommender_object.RECOMMENDER_NAME, i)
        # recommender_object.save_model(output_root_path, file_name=saved_name)

    avg_MAP = sum(MAP_list) / k
    diff_MAP = max(MAP_list) - min(MAP_list)
    print("Average {}-fold MAP: {:.7f}, diff: {:.7f}".format(k, avg_MAP, diff_MAP))
    logFile.write("Average {}-fold MAP: {:.7f}, diff: {:.7f}".format(k, avg_MAP, diff_MAP))
    logFile.flush()


def evaluate_all_ICMs(URM_all):
    ICM_channel = load_icm("data_ICM_channel.csv", weight=1)
    ICM_event = load_icm("data_ICM_event.csv", weight=1)
    ICM_genre = load_icm("data_ICM_genre.csv", weight=1)
    ICM_subgenre = load_icm("data_ICM_subgenre.csv", weight=1)
    ICM_all = sps.hstack([ICM_genre, ICM_subgenre, ICM_channel, ICM_event]).tocsr()
    ICMs = [None, ICM_channel, ICM_event, ICM_genre, ICM_subgenre, ICM_all]
    names = ['NO ICM', 'ICM_channel', 'ICM_event', 'ICM_genre', 'ICM_subgenre', 'ICM_all']

    for name, ICM in zip(names, ICMs):
        print('Using ' + name)
        evaluate_all_recommenders(URM_all, ICM)


if __name__ == '__main__':
    URM_all = load_urm()
    # evaluate_kfold(k=5)
    evaluate_all_recommenders(URM_all)
    # evaluate_best_saved_model(URM_all)
    # evaluate_all_ICMs(URM_all)
