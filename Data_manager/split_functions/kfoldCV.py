from Evaluation.Evaluator import EvaluatorHoldout
import scipy.sparse as sps
import numpy as np


def split_kfold(URM_all, k=5):
    numInteractions = URM_all.nnz
    URM_all = URM_all.tocoo()
    shape = URM_all.shape

    URM_train_list = []
    URM_valid_list = []

    numInteractions_validation = int(numInteractions / k)

    random_indices = np.random.permutation(numInteractions)

    for i in range(k):
        valid_mask = random_indices[i * numInteractions_validation:(i + 1) * numInteractions_validation]

        train_mask = []

        for index in random_indices:
            if index not in valid_mask:
                train_mask.append(index)

        train_mask = np.array(train_mask)

        URM_train = sps.coo_matrix((URM_all.data[train_mask], (URM_all.row[train_mask], URM_all.col[train_mask])),
                                   shape=shape)
        URM_train = URM_train.tocsr()

        URM_valid = sps.coo_matrix((URM_all.data[valid_mask], (URM_all.row[valid_mask], URM_all.col[valid_mask])),
                                   shape=shape)
        URM_valid = URM_valid.tocsr()

        URM_train_list.append(URM_train)
        URM_valid_list.append(URM_valid)

    evaluator_list = []

    for URM_validation in URM_valid_list:
        evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=[10])
        evaluator_list.append(evaluator_validation)

    return evaluator_list, URM_train_list, URM_valid_list
