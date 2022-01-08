#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 22/11/17

@author: Maurizio Ferrari Dacrema
"""
import multiprocessing
import os
from functools import partial
from multiprocessing.pool import ThreadPool as Pool1
import scipy.sparse as sps

from Utils.split_train_validation_random_holdout import \
    split_train_in_two_percentage_global_sample
from Evaluation.Evaluator import EvaluatorHoldout
from HyperparameterTuning.run_hyperparameter_search import runHyperparameterSearch_Collaborative, \
    runHyperparameterSearch_Content, runHyperparameterSearch_Hybrid
from Recommenders.Hybrids.HybridRatings_IALS_hybrid_EASE_R_hybrid_SLIM_Rp3 import \
    HybridRatings_IALS_hybrid_EASE_R_hybrid_SLIM_Rp3
from Recommenders.KNN.ItemKNNCBFWeightedSimilarityRecommender import ItemKNNCBFWeightedSimilarityRecommender
from Recommenders.Recommender_import_list import *
from reader import load_urm, load_icm, get_k_folds_URM

output_folder_path = "result_experiments/"

# If directory does not exist, create
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)


def read_data_split_and_search():
    """
    This function provides a simple example on how to tune parameters of a given algorithm

    The BayesianSearch object will save:
        - A .txt file with all the cases explored and the recommendation quality
        - A _best_model file which contains the trained model and can be loaded with recommender.load_model()
        - A _best_parameter file which contains a dictionary with all the fit parameters, it can be passed to recommender.fit(**_best_parameter)
        - A _best_result_validation file which contains a dictionary with the results of the best solution on the validation
        - A _best_result_test file which contains a dictionary with the results, on the test set, of the best solution chosen using the validation set
    """

    collaborative_algorithm_list = [
        # P3alphaRecommender,
        # RP3betaRecommender,
        # ItemKNNCFRecommender,
        # UserKNNCFRecommender,
        # MatrixFactorization_BPR_Cython,  # bad
        # MatrixFactorization_FunkSVD_Cython,
        # PureSVDRecommender,
        # SLIM_BPR_Cython,
        # SLIM_BPR_Cython
        # SLIMElasticNetRecommender,
        # IALSRecommender
        # MultVAERecommender
        # IALSRecommender_implicit
        # EASE_R_Recommender
    ]

    content_algorithm_list = [
        ItemKNNCBFRecommender,
        # ItemKNNCBFWeightedSimilarityRecommender,
    ]

    hybrid_algorithm_list = [
        # ScoresHybridRecommender,
        # HybridWsparseSLIMRp3,
        # Hybrid_SlimElastic_Rp3_IALS,
        # ItemKNNScoresHybridRecommender
        # RankingHybrid

        # Hybrid_SlimElastic_Rp3_PureSVD
        # HybridSimilarity_withGroupedusers
        # Hybrid_SLIM_EASE_R_IALS
        # HybridRatings_EASE_R_hybrid_SLIM_Rp3
        HybridRatings_IALS_hybrid_EASE_R_hybrid_SLIM_Rp3
    ]

    URM_all = load_urm()

    URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_all, train_percentage=0.80)

    cutoff_list = [10]
    metric_to_optimize = "MAP"
    cutoff_to_optimize = 10

    n_cases = 100
    n_random_starts = int(n_cases / 3)

    # new function to evaluate 1 group of users (for now split at 50%)
    # evaluator_validation = group_users_in_urm(URM_train, URM_validation, 1)

    evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=cutoff_list)

    # COLLABORATIVE
    # runParameterSearch_Collaborative_partial = partial(runHyperparameterSearch_Collaborative,
    #                                                    URM_train=URM_train,
    #                                                    metric_to_optimize=metric_to_optimize,
    #                                                    cutoff_to_optimize=cutoff_to_optimize,
    #                                                    n_cases=n_cases,
    #                                                    n_random_starts=n_random_starts,
    #                                                    evaluator_validation_earlystopping=evaluator_validation,
    #                                                    evaluator_validation=evaluator_validation,
    #                                                    evaluator_test=None,
    #                                                    output_folder_path=output_folder_path,
    #                                                    resume_from_saved=True,
    #                                                    save_model="no",
    #                                                    similarity_type_list=None,  # ["cosine"],
    #                                                    allow_weighting=True,
    #                                                    parallelizeKNN=False)
    #
    # pool_collab = multiprocessing.Pool(processes=int(multiprocessing.cpu_count()), maxtasksperchild=1)
    # pool_collab.map(runParameterSearch_Collaborative_partial, collaborative_algorithm_list)

    # CONTENT RECS

    ICM_channel = load_icm("data_ICM_channel.csv")
    # ICM_event = load_icm("data_ICM_event.csv")
    # ICM_genre = load_icm("data_ICM_genre.csv")
    # ICM_subgenre = load_icm("data_ICM_subgenre.csv")
    # ICM_all = sps.hstack([ICM_channel, ICM_event, ICM_genre, ICM_subgenre]).tocsr()
    #
    # runParameterSearch_Content_partial = partial(runHyperparameterSearch_Content,
    #                                              URM_train=URM_train,
    #                                              ICM_object=ICM_all,
    #                                              ICM_name="ICM_all",
    #                                              metric_to_optimize=metric_to_optimize,
    #                                              cutoff_to_optimize=cutoff_to_optimize,
    #                                              n_cases=n_cases,
    #                                              n_random_starts=n_random_starts,
    #                                              evaluator_validation=evaluator_validation,
    #                                              evaluator_test=None,
    #                                              output_folder_path=output_folder_path,
    #                                              resume_from_saved=False,
    #                                              save_model="no",
    #                                              similarity_type_list=None,
    #                                              parallelizeKNN=False)
    #
    # pool_collab = Pool1(processes=int(multiprocessing.cpu_count()))
    # pool_collab.map(runParameterSearch_Content_partial, content_algorithm_list)

    # HYBRID
    runParameterSearch_Hybrid_partial = partial(runHyperparameterSearch_Hybrid,
                                                URM_train=URM_train,
                                                W_train=None,
                                                ICM_object=ICM_channel,
                                                ICM_name="ICM_all",
                                                metric_to_optimize=metric_to_optimize,
                                                cutoff_to_optimize=cutoff_to_optimize,
                                                n_cases=n_cases,
                                                n_random_starts=n_random_starts,
                                                evaluator_validation_earlystopping=evaluator_validation,
                                                evaluator_validation=evaluator_validation,
                                                evaluator_test=None,
                                                output_folder_path=output_folder_path)

    pool_collab = Pool1(processes=int(multiprocessing.cpu_count()))
    pool_collab.map(runParameterSearch_Hybrid_partial, hybrid_algorithm_list)

if __name__ == '__main__':
    read_data_split_and_search()
