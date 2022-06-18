from data_gen import *
from anneal import Anneal

def main():
    # Dataset related constants
    N_c1 = 100 # Number of users in cluster 1
    N_c2 = 150 # Number of users in cluster 2
    N_u = 250 # Number of uniformly distributed users

    N_ue = N_c1 + N_c2 + N_u # Total number of users
    N_s = 200 # number of possible BS sites

    max_u = 10000

    ue = rand_coords(N_c1, N_c2, N_u, max_u=max_u) # user coordinates with default mu & sigma values
    S = rand_uniform(N_s, max_u) # possible BS sites

    # Add columns for ue / bs status (ue -> served / not, bs -> in use / not)
    ue = np.c_[ue, np.full((ue.shape[0], 1), False)]
    S = np.c_[S, np.full((S.shape[0], 1), False)]

    B_max = 30 # Number of deployed BSs
    R_ue = 1 # Individual UE revenue
    C_bs = 5 # BS operational cost

    SA = Anneal(ue, S, B_max, R_ue, C_bs)

    print(SA.user_coords)
    SA.users_within_radius(S[0,0], S[0,1])


main()