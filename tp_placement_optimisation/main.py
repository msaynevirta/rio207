from data_gen import *
from anneal import Anneal
import matplotlib.pyplot as plt

def main():
    # Dataset related constants
    N_c1 = 100 # Number of users in cluster 1
    N_c2 = 150 # Number of users in cluster 2
    N_u = 250 # Number of uniformly distributed users

    N_ue = N_c1 + N_c2 + N_u # Total number of users
    N_s = 200 # number of possible BS sites

    B_max = 30 # Number of deployed BSs
    R_ue = 1 # Individual UE revenue
    C_bs = 5 # BS operational cost

    max_u = 10000 # dimensions km
    R_cell = 954 # cell radius in meters
    h_bs = 30 # cell tower height in meters
    h_ue = 1.5 # ue height in meters

    ue = rand_coords(N_c1, N_c2, N_u, max_u=max_u) # user coordinates with default mu & sigma values
    S = rand_uniform(N_s, max_u) # possible BS sites

    # Add columns for ue / bs status (ue -> bs index serving / -1, bs -> in use / not)
    ue = np.c_[ue, np.full((ue.shape[0], 1), False)]
    S = np.c_[S, np.full((S.shape[0], 1), False)]

    opt_params = { 'B_max' :  B_max,
                   'R_ue' :   R_ue,
                   'C_bs' :    C_bs }

    ue_bs_params = { 'R_cell' :  954,
                     'h_bs' :   30.0,
                     'h_ue' :    1.5,
                     'P_tx_bs' :  46,
                     'G_ant_bs' : 19 }

    SA = Anneal(ue, S, opt_params, ue_bs_params)
    SA.anneal()
    
    fig, (ax1, ax2) = plt.subplots(2)
    SA.plot_energy(ax1)
    SA.plot_scatter_bs_ue(ax2)
    plt.show()


main()