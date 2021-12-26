import os
from datetime import datetime

from Recommenders.Recommender_import_list import *
from reader import load_urm, load_target

res_dir = 'result_experiments/csv'
output_root_path = "./result_experiments/"

# If directory does not exist, create
if not os.path.exists(output_root_path):
    os.makedirs(output_root_path)

logFile = open(output_root_path + "submission_all_algorithms.txt", "a")


def create_csv(target_ids, results, rec_name):
    exp_dir = os.path.join(res_dir, rec_name)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    csv_fname = 'results_' + datetime.now().strftime('%b%d_%H-%M-%S') + '.csv'

    with open(os.path.join(exp_dir, csv_fname), 'w') as f:
        f.write('user_id,item_list\n')
        for target_id, result in zip(target_ids, results):
            f.write(str(target_id) + ', ' + ' '.join(map(str, result)) + '\n')


def run_prediction_all_recommenders(URM_all, target_ids):
    recommender_object = SLIMElasticNetRecommender(URM_all)

    fit_params = {}
    recommender_object.fit(**fit_params)
    # recommender_object.save_model(output_root_path, file_name="hybridsimilarityslimrp3withstack.zip")

    item_list = recommender_object.recommend(target_ids, cutoff=10)
    create_csv(target_ids, item_list, recommender_object.RECOMMENDER_NAME)


# Method to create prediction using a saved best model of a specific recommender class

def run_prediction_best_saved_model(URM_all, ICM=None):
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
    URM_all, user_id_unique, item_id_unique = load_urm()
    target_ids = load_target()

    # ICM_channel = load_icm("data_ICM_channel.csv", weight=1)
    # ICM_event = load_icm("data_ICM_event.csv", weight=1)
    # ICM_genre = load_icm("data_ICM_genre.csv", weight=1)
    # ICM_subgenre = load_icm("data_ICM_subgenre.csv", weight=1)
    # ICM_all = sps.hstack([ICM_genre, ICM_subgenre, ICM_channel, ICM_event]).tocsr()
    # ICMs = [ICM_genre, ICM_subgenre, ICM_channel, ICM_event, ICM_all]

    run_prediction_all_recommenders(URM_all, target_ids)
    # run_prediction_best_saved_model(URM_all)
