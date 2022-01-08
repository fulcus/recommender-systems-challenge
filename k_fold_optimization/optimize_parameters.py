import os
import pandas as pd
import skopt
from skopt.utils import use_named_args

import k_fold_optimization.evaluate
import k_fold_optimization.dataset
from Recommenders.Hybrids.HybridRatings_IALS_hybrid_EASE_R_hybrid_SLIM_Rp3 import \
    HybridRatings_IALS_hybrid_EASE_R_hybrid_SLIM_Rp3
from k_fold_optimization.hyperparam_def import names, spaces

output_root_path = "./optimization_data/"

# If directory does not exist, create
if not os.path.exists(output_root_path):
    os.makedirs(output_root_path)


def load_df(name):
    filename = output_root_path + name + ".res"
    if os.path.exists(filename):
        df = pd.read_pickle(filename)
        return df
    else:
        return None


def read_df(name, param_names, metric="MAP"):
    df = load_df(name)

    if df is not None:
        y = df[metric].tolist()
        x_series = [df[param_name].tolist() for param_name in param_names]
        x = [t for t in zip(*x_series)]

        return x, y
    else:
        return None, None


def store_df(name, df: pd.DataFrame):
    filename = output_root_path + name + ".res"
    df.to_pickle(filename)


def append_new_data_to_df(name, new_df):
    df = load_df(name)
    df = df.append(new_df, ignore_index=True)
    store_df(name, df)


def create_df(param_tuples, param_names, value_list, metric="MAP"):
    df = pd.DataFrame(data=param_tuples, columns=param_names)
    df[metric] = value_list
    return df


def optimize_parameters(URMrecommender_class: type, n_calls=100, k=5, validation_percentage=0.05, n_random_starts=None,
                        seed=None, limit_at=1000, forest=False, xi=0.01):
    if n_random_starts is None:
        n_random_starts = int(0.5 * n_calls)

    name = names[URMrecommender_class]
    space = spaces[URMrecommender_class]

    if validation_percentage > 0:
        print("Using randomized datasets. k={}, val_percentage={}".format(k, validation_percentage))
        URM_trains, URM_tests, ICM_trains = k_fold_optimization.dataset.give_me_randomized_k_folds_with_val_percentage(k,
                                                                                                                       validation_percentage)
    else:
        print("Splitting original datasets in N_folds:{}".format(k))
        URM_trains, URM_tests, ICM_trains = k_fold_optimization.dataset.give_me_k_folds(k)

    if len(URM_trains) > limit_at:
        URM_trains = URM_trains[:limit_at]
        URM_tests = URM_tests[:limit_at]
        ICM_trains = ICM_trains[:limit_at]

    assert (len(URM_trains) == len(URM_tests) and len(URM_tests) == len(ICM_trains))
    print("Starting optimization: N_folds={}, slim_name={}".format(len(URM_trains), names[URMrecommender_class]))

    if URMrecommender_class == HybridRatings_IALS_hybrid_EASE_R_hybrid_SLIM_Rp3:
        recommenders = []
        for i, URM_train_csr in enumerate(URM_trains):
            recommenders.append(URMrecommender_class(URM_train_csr, i))

        @use_named_args(space)
        def objective(**params):
            scores = []
            for recommender, test in zip(recommenders, URM_tests):
                recommender.fit(**params)
                _, _, MAP = k_fold_optimization.evaluate.evaluate_algorithm(test, recommender)
                scores.append(-MAP)
            print("Just Evaluated this: {}".format(params))
            return sum(scores) / len(scores)

    else:
        @use_named_args(space)
        def objective(**params):
            scores = []
            for URM_train_csr, test in zip(URM_trains, URM_tests):
                recommender = URMrecommender_class(URM_train_csr, **params)
                recommender.fit()
                _, _, MAP = k_fold_optimization.evaluate.evaluate_algorithm(test, recommender)
                scores.append(-MAP)
                print("MAP: {}".format(MAP))

            print("Just Evaluated this: {}".format(params))
            print("MAP: {}, diff: {}".format(sum(scores) / len(scores), max(scores) - min(scores)))

            return sum(scores) / len(scores)

    # xs, ys = _load_xy(slim_name)
    param_names = [v.name for v in spaces[URMrecommender_class]]
    xs, ys = read_df(name, param_names)

    if not forest:
        res_gp = skopt.gp_minimize(
            objective,
            space,
            n_calls=n_calls,
            n_random_starts=n_random_starts,
            n_points=10000,
            n_jobs=1,
            # noise = 'gaussian',
            noise=1e-5,
            acq_func='gp_hedge',
            acq_optimizer='auto',
            random_state=None,
            verbose=True,
            n_restarts_optimizer=10,
            xi=xi,
            kappa=1.96,
            x0=xs,
            y0=ys,
        )
    else:
        res_gp = skopt.forest_minimize(
            objective,
            space,
            n_calls=n_calls,
            n_random_starts=n_random_starts,
            verbose=True,
            x0=xs,
            y0=ys,
            acq_func="EI",
            xi=xi
        )

    print("Writing a total of {} points for {}. Newly added records: {}".format(len(res_gp.x_iters), name,
                                                                                n_calls))

    # _store_xy(slim_name, res_gp.x_iters, res_gp.func_vals)
    df = create_df(res_gp.x_iters, param_names, res_gp.func_vals, "MAP")
    store_df(names[URMrecommender_class], df)

    print(name + " reached best performance = ", -res_gp.fun, " at: ", res_gp.x)
