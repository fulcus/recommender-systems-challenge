#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14/09/17

@author: Maurizio Ferrari Dacrema
"""

import pandas as pd
from Data_manager.DataReader import DataReader
from Data_manager.DatasetMapperManager import DatasetMapperManager
from Data_manager.TVShows._utils_tvshows_parser import _loadURM, _loadICM


class TVShowsReader(DataReader):

    DATASET_SUBFOLDER = "TVShows/"
    AVAILABLE_URM = ["URM_all"]
    AVAILABLE_ICM = ["ICM_channel", "ICM_event", "ICM_genre", "ICM_subgenre"]

    IS_IMPLICIT = True

    def _get_dataset_name_root(self):
        return self.DATASET_SUBFOLDER

    def _load_from_original_file(self):
        # Load data from original
        dataset_path = self.DATASET_SPLIT_ROOT_FOLDER + self.DATASET_SUBFOLDER

        URM_path = dataset_path + "data_train.csv"
        ICM_channel_path = dataset_path + "data_ICM_channel.csv"
        ICM_event_path = dataset_path + "data_ICM_event.csv"
        ICM_genre_path = dataset_path + "data_ICM_genre.csv"
        ICM_subgenre_path = dataset_path + "data_ICM_subgenre.csv"

        self._print("Loading Interactions")
        URM_all_dataframe = _loadURM(URM_path)

        self._print("Loading Item Features")
        ICM_channel_df = _loadICM(ICM_channel_path, feature="Channel")
        ICM_event_df = _loadICM(ICM_event_path, feature="Event")
        ICM_genre_df = _loadICM(ICM_genre_path, feature="Genre")
        ICM_subgenre_df = _loadICM(ICM_subgenre_path, feature="Subgenre")

        dataset_manager = DatasetMapperManager()
        dataset_manager.add_URM(URM_all_dataframe, "URM_all")
        dataset_manager.add_ICM(ICM_channel_df, "ICM_channel")
        dataset_manager.add_ICM(ICM_event_df, "ICM_event")
        dataset_manager.add_ICM(ICM_genre_df, "ICM_genre")
        dataset_manager.add_ICM(ICM_subgenre_df, "ICM_subgenre")

        loaded_dataset = dataset_manager.generate_Dataset(dataset_name=self._get_dataset_name(),
                                                          is_implicit=self.IS_IMPLICIT)

        self._print("Loading Complete")

        return loaded_dataset
