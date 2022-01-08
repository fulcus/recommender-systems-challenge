import os
from datetime import datetime

from Recommenders.Hybrids.HybridRatings_IALS_hybrid_EASE_R_hybrid_SLIM_Rp3 import \
    HybridRatings_IALS_hybrid_EASE_R_hybrid_SLIM_Rp3
from Recommenders.Recommender_import_list import *
from evaluate import _get_params
from reader import load_urm, load_target, load_icm

res_dir = 'result_experiments/csv'
output_root_path = "./result_experiments/"

# If directory does not exist, create
if not os.path.exists(output_root_path):
    os.makedirs(output_root_path)


def create_csv(target_ids, results, rec_name):
    exp_dir = os.path.join(res_dir, rec_name)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    csv_fname = 'results_' + datetime.now().strftime('%b%d_%H-%M-%S') + '.csv'

    with open(os.path.join(exp_dir, csv_fname), 'w') as f:
        f.write('user_id,item_list\n')
        for target_id, result in zip(target_ids, results):
            f.write(str(target_id) + ', ' + ' '.join(map(str, result)) + '\n')


def run_prediction_on_target(URM_all, target_ids):
    recommender_object = HybridRatings_IALS_hybrid_EASE_R_hybrid_SLIM_Rp3(URM_all)
    fit_params = _get_params(recommender_object)
    recommender_object.fit(**fit_params)

    item_list = recommender_object.recommend(target_ids, cutoff=10)
    create_csv(target_ids, item_list, recommender_object.RECOMMENDER_NAME)


def run_prediction_best_saved_model(URM_all, ICM=None):
    """
    Create prediction using a saved best model of a specific recommender class
    """
    # ******** set here the recommender you want to use
    # recommender_object = SLIMElasticNetRecommender(URM_all)
    recommender_object = EASE_R_Recommender(URM_all)

    # rec_best_model_last.zip is the output of the run_hyperparameter_search (one best model for each rec class)
    # recommender_object.load_model(output_root_path, file_name=recommender_object.RECOMMENDER_NAME + "_best_model.zip")

    # recommender_object.load_model(output_root_path, file_name="slimelastic_urmall_453.zip")
    recommender_object.load_model(output_root_path, file_name="EASE_R_Recommender_best_model.zip")

    # added for prediction
    item_list = recommender_object.recommend(target_ids, cutoff=10, remove_seen_flag=True)
    create_csv(target_ids, item_list, recommender_object.RECOMMENDER_NAME)


if __name__ == '__main__':
    URM_all = load_urm()
    target_ids = load_target()

    # ICM_channel = load_icm("data_ICM_channel.csv", weight=1)
    # ICM_event = load_icm("data_ICM_event.csv", weight=1)
    # ICM_genre = load_icm("data_ICM_genre.csv", weight=1)
    # ICM_subgenre = load_icm("data_ICM_subgenre.csv", weight=1)
    # ICM_all = sps.hstack([ICM_genre, ICM_subgenre, ICM_channel, ICM_event]).tocsr()
    # ICMs = [ICM_genre, ICM_subgenre, ICM_channel, ICM_event, ICM_all]

    run_prediction_on_target(URM_all, target_ids)
    # run_prediction_best_saved_model(URM_all)
