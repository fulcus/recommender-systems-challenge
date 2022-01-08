from Recommenders.Hybrids.HybridRatings_IALS_hybrid_EASE_R_hybrid_SLIM_Rp3 import \
    HybridRatings_IALS_hybrid_EASE_R_hybrid_SLIM_Rp3
from k_fold_optimization.optimize_parameters import optimize_parameters

if __name__ == '__main__':
    val_percentage = 0
    k = 5

    rec_class = HybridRatings_IALS_hybrid_EASE_R_hybrid_SLIM_Rp3
    optimize_parameters(
        URMrecommender_class=rec_class,
        validation_percentage=val_percentage,
        n_random_starts=1,
        k=k,
        n_calls=5,
        limit_at=10,
        forest=True,
        xi=0.001
    )
