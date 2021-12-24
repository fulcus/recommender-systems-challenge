"""
Created on 23/03/2019
@author: Maurizio Ferrari Dacrema
"""

import implicit
import numpy as np

from Recommenders.BaseMatrixFactorizationRecommender import BaseMatrixFactorizationRecommender


class IALSRecommender_implicit(BaseMatrixFactorizationRecommender):
    """
    ALS implemented with implicit following guideline of
    https://medium.com/radon-dev/als-implicit-collaborative-filtering-5ed653ba39fe
    IDEA:
    Recomputing x_{u} and y_i can be done with Stochastic Gradient Descent, but this is a non-convex optimization problem.
    We can convert it into a set of quadratic problems, by keeping either x_u or y_i fixed while optimizing the other.
    In that case, we can iteratively solve x and y by alternating between them until the algorithm converges.
    This is Alternating Least Squares.
    """

    RECOMMENDER_NAME = "IALSRecommender_implicit"

    def __init__(self, URM_train, verbose=True):
        super(IALSRecommender_implicit, self).__init__(URM_train, verbose=verbose)

    def fit(self, n_factors=50, regularization=0.001847510119137634, iterations=30, num_threads=2):
        self.n_factors = n_factors
        self.regularization = regularization
        self.iterations = iterations

        sparse_item_user = self.URM_train.T

        # Initialize the als model and fit it using the sparse item-user matrix
        model = implicit.als.AlternatingLeastSquares(factors=self.n_factors, regularization=self.regularization,
                                                     iterations=self.iterations, num_threads=num_threads)

        alpha_val = 2
        # Calculate the confidence by multiplying it by our alpha value.

        data_conf = (sparse_item_user * alpha_val).astype('double')

        # Fit the model
        model.fit(data_conf)

        # Get the user and item vectors from our trained model
        self.USER_factors = model.user_factors
        self.ITEM_factors = model.item_factors

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        """
        USER_factors is n_users x n_factors
        ITEM_factors is n_items x n_factors

        The prediction for cold users will always be -inf for ALL items

        :param user_id_array:
        :param items_to_compute:
        :return:
        """

        assert self.USER_factors.shape[1] == self.ITEM_factors.shape[1], \
            "{}: User and Item factors have inconsistent shape".format(self.RECOMMENDER_NAME)

        assert self.USER_factors.shape[0] > np.max(user_id_array), \
            "{}: Cold users not allowed. Users in trained model are {}, requested prediction for users up to {}".format(
                self.RECOMMENDER_NAME, self.USER_factors.shape[0], np.max(user_id_array))

        if items_to_compute is not None:
            item_scores = - np.ones((len(user_id_array), self.ITEM_factors.shape[0]), dtype=np.float32) * np.inf
            item_scores[:, items_to_compute] = np.dot(self.USER_factors[user_id_array],
                                                      np.transpose(self.ITEM_factors[items_to_compute, :]))

        else:
            item_factors_T = np.transpose(self.ITEM_factors)
            user_factors = self.USER_factors[user_id_array]
            item_scores = np.dot(user_factors, item_factors_T)

        # No need to select only the specific negative items or warm users because the -inf score will not change
        if self.use_bias:
            item_scores += self.ITEM_bias + self.GLOBAL_bias
            item_scores = np.transpose(np.transpose(item_scores) + self.USER_bias[user_id_array])

        return item_scores
