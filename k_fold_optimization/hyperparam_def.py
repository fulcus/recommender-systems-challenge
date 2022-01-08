from skopt.space import Real, Integer, Categorical

from Recommenders.Hybrids.HybridRatings_IALS_hybrid_EASE_R_hybrid_SLIM_Rp3 import \
    HybridRatings_IALS_hybrid_EASE_R_hybrid_SLIM_Rp3

names = {}
spaces = {}

names[HybridRatings_IALS_hybrid_EASE_R_hybrid_SLIM_Rp3] = "HybridRatings_IALS_hybrid_EASE_R_hybrid_SLIM_Rp3"
spaces[HybridRatings_IALS_hybrid_EASE_R_hybrid_SLIM_Rp3] = [
    Real(low=0., high=1., prior='uniform', name='alpha'),
    # Real(low=0., high=1., prior='uniform', slim_name='alpha1'),
    Real(low=0., high=1., prior='uniform', name='beta'),
    # Real(low=0., high=1., prior='uniform', slim_name='beta1'),
    Real(low=0., high=1., prior='uniform', name='gamma')
]
