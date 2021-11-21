#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 26/11/19

@author: Maurizio Ferrari Dacrema
"""

import pandas as pd
import numpy as np

def _loadICM(path, feature):
    ICM = pd.read_csv(filepath_or_buffer=path, sep=',', header=0, dtype={0: str, 1: str, 2: np.int32}, engine='python')
    ICM.columns = ["ItemID", feature, "Data"]
    ICM["FeatureID"] = feature
    return ICM

def _loadURM(URM_path):
    URM_all_dataframe = pd.read_csv(filepath_or_buffer=URM_path, sep=',', header=0, dtype={0: str, 1: str, 2: np.int32}, engine='python')
    URM_all_dataframe.columns = ["UserID", "ItemID", "Data"]
    return URM_all_dataframe

def load_target(path):
    df_original = pd.read_csv(filepath_or_buffer="path", sep=',', header=0, dtype={'user_id': np.int32})

    df_original.columns = ['user']
    user_id_list = df_original['user'].values
    user_id_unique = np.unique(user_id_list)
    return user_id_unique

def _loadICM_tags(tags_path, header=True, separator=','):

    # Tags
    from Data_manager.TagPreprocessing import tagFilterAndStemming

    fileHandle = open(tags_path, "r", encoding="latin1")

    if header is not None:
        fileHandle.readline()

    movie_id_list = []
    tags_lists = []

    for index, line in enumerate(fileHandle):

        if index % 100000 == 0 and index>0:
            print("Processed {} rows".format(index))

        if (len(line)) > 1:
            line = line.split(separator)

            line[-1] = line[-1].replace("\n", "")

            # If a movie has no genre, ignore it
            movie_id = line[1]
            this_tag_list = line[2]

            # Remove non alphabetical character and split on spaces
            this_tag_list = tagFilterAndStemming(this_tag_list)

            movie_id_list.append(movie_id)
            tags_lists.append(this_tag_list)

    fileHandle.close()

    ICM_dataframe = pd.DataFrame(tags_lists, index=movie_id_list).stack()
    ICM_dataframe = ICM_dataframe.reset_index()[["level_0", 0]]
    ICM_dataframe.columns = ['ItemID', 'FeatureID']
    ICM_dataframe["Data"] = 1


    return ICM_dataframe





def _loadUCM(UCM_path, header=True, separator=','):

    # Genres
    from Data_manager.IncrementalSparseMatrix import IncrementalSparseMatrix_FilterIDs

    ICM_builder = IncrementalSparseMatrix_FilterIDs(preinitialized_col_mapper = None, on_new_col = "add",
                                                    preinitialized_row_mapper = None, on_new_row = "add")


    fileHandle = open(UCM_path, "r", encoding="latin1")
    numCells = 0

    if header:
        fileHandle.readline()

    for line in fileHandle:
        numCells += 1
        if (numCells % 1000000 == 0):
            print("Processed {} rows".format(numCells))

        if (len(line)) > 1:
            line = line.split(separator)

            line[-1] = line[-1].replace("\n", "")

            user_id = line[0]

            token_list = []
            token_list.append("gender_" + str(line[1]))
            token_list.append("age_group_" + str(line[2]))
            token_list.append("occupation_" + str(line[3]))
            token_list.append("zip_code_" + str(line[4]))

            # Rows movie ID
            # Cols features
            ICM_builder.add_single_row(user_id, token_list, data = 1.0)


    fileHandle.close()

    return ICM_builder.get_SparseMatrix(), ICM_builder.get_column_token_to_id_mapper(), ICM_builder.get_row_token_to_id_mapper()






