# recommender-systems-challenge

[![Kaggle](https://img.shields.io/badge/open-kaggle-blue)](https://www.kaggle.com/c/recommender-system-2021-challenge-polimi)

This repository contains the source code for the 2020 Polimi Recommender System Challenge on [kaggle](https://www.kaggle.com/c/recommender-system-2021-challenge-polimi).

The goal of the competition was to create the recommender system for TV programs by providing 10 recommended products to each user. 

## Data

Given the User Rating Matrix and four Item Content Matrices we had to recommend 10 relevant tv shows to the users. 

The URM contained 5M interactions, 13650 users and 18059 item and a sparsity of 97.86 %.

The ICMs contained information about the channels, episodes, genre and subgenre of the shows.

All data was anonymized, so it was not possible to perform text analysis, genre grouping, correlation between text and popularity and so on.

## Recommender

Our final recommender was a hybrid, obtained combining of the following models:
* SLIM ElasticNet
* RP3Beta 
* EASE R 
* Implicit Alternating Least Squares 

The final hybrid was obtained as a linear combination of the ratings of EASE R + IALS + hybrid(SLIM + RP3Beta).

The hybrid combining SLIM and RP3Beta merges their similarity matrices with a weighted sum. 

## Evaluation
The evaluation metric was MAP@10.
* Public leaderboard score: 0.48575 (8th)
* Private leaderboard score: 0.48656 (7th)

## Credits
This repository contains code from the [course framework repo](https://github.com/MaurizioFD/RecSys_Course_AT_PoliMi), that provides recommender implementations and utility code. The k_fold_optimization
code was taken from [this repo](https://github.com/LCarmi/recommender-systems-2020-challenge-polimi).

## Team
[Arianna Galzerano](https://github.com/arigalzi) & [Francesco Gonzales](https://github.com/fulcus)
