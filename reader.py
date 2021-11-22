import pandas as pd
import scipy.sparse as sps
import numpy as np

urm_path = "Data_manager_split_datasets/TVShows/data_train.csv"
icm_path = "Data_manager_split_datasets/TVShows/"
target_path = "Data_manager_split_datasets/TVShows/data_target_users_test.csv"


def load_urm():
    df_original = pd.read_csv(filepath_or_buffer=urm_path, sep=',', header=0,
                              dtype={0: np.int32, 1: np.int32, 2: np.int32})

    df_original.columns = ["UserID", "ItemID", "Data"]

    user_id_list = df_original['UserID'].values
    item_id_list = df_original['ItemID'].values
    rating_id_list = df_original['Data'].values

    user_id_unique = np.unique(user_id_list)
    item_id_unique = np.unique(item_id_list)

    csr_matrix = sps.csr_matrix((rating_id_list, (user_id_list, item_id_list)))
    csr_matrix = csr_matrix.astype(dtype=np.int32)
    # print("DataReader:")
    # print("\tLoading the URM:")
    # print("\t\tURM size:" + str(csr_matrix.shape))
    # print("\t\tURM unique users:" + str(user_id_unique.size))
    # print("\t\tURM unique items:" + str(item_id_unique.size))
    # print("\tURM loaded.")

    return csr_matrix, user_id_unique, item_id_unique


def load_target():
    df_original = pd.read_csv(filepath_or_buffer=target_path, sep=',', header=0,
                              dtype={'UserID': np.int32})

    df_original.columns = ['UserID']

    user_id_list = df_original['UserID'].values

    user_id_unique = np.unique(user_id_list)

    # print("DataReader:")
    # print("\tLoading the target users:")
    # print("\t\tTarget size:" + str(user_id_unique.shape))
    # print("\tTarget users loaded.")

    return user_id_unique


def load_icm(icm_file, weight=1):
    df_original = pd.read_csv(filepath_or_buffer=icm_path + icm_file, sep=',', header=0,
                              dtype={'ItemID': np.int32, 'Feature': np.int32, 'Data': np.int32})

    df_original.columns = ['ItemID', 'Feature', 'Data']

    item_id_list = df_original['ItemID'].values
    feature_id_list = df_original['Feature'].values
    data_id_list = df_original['Data'].values * weight

    csr_matrix = sps.csr_matrix((data_id_list, (item_id_list, feature_id_list)))

    # print("DataReader:")
    # print("\tLoading the asset ICM: " + icm_asset_path)
    # print("\t\tAsset ICM size:" + str(csr_matrix.shape))
    # print("\tAsset ICM loaded.")

    return csr_matrix


# to delete
def load_urm_icm():
    urm, _, _ = load_urm()
    icm = load_icm("data_ICM_subgenre.csv", 1)
    urm_icm = sps.vstack([urm, icm.T])
    urm_icm = urm_icm.tocsr()

    return urm_icm
