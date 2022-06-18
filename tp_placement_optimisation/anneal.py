import numpy as np
import matplotlib.pyplot as plt

class Anneal(object):
    def __init__(self, user_coords, bs_sites, B_max, R_ue, C_bs):
        """
        Class for simulated anneal
        Parameters
        ----------
        user_coords : np.array of 2D user coordinates
        bs_sites : np.array of 2D bs sites
        N_ue : total number of users
        N_s : number of possible BS sites
        B_max : max amount of deployed BSs
        R_ue : revenue per UE
        C_bs : operating cost per BS
        """
        self.user_coords = user_coords # UE locations and status (served / not)
        self.bs_sites = bs_sites # possible BS sites and status (in use / not)
        self.N_bs = 0 # number of chosen BS sites

        self.B_max = B_max # Max number of deployed BSs
        self.R_ue = R_ue # Individual UE revenue
        self.C_bs = C_bs # BS operational cost

        self.T = 5000 # starting temperature
        self.alpha = 0.995
        self.stopping_temperature = 1e-8
        self.stopping_iter = 100000
        self.iteration = 1

        self.best_solution = None
        self.highest_energy = float("Inf")
        self.energy_list = []

    def add_bs():
        """
        A free BS site is chosen at random and a BS is placed at this location.
        Moves the coordinates of a certain BS site from self.bs_sites to
        self.bs_choices.
        """
        pass

    def remove_bs():
        """
        A free BS site is chosen at random and a BS is placed at this location.
        Returns the coordinates of a certain BS site from self.bs_choices to
        self.bs_sites.
        """
        pass

    def move_bs():
        """
        A free BS site is chosen at random and a BS is placed at this location.
        Swaps two sets of coordinates between self.bs_choices and self.bs_sites.
        """
        pass

    def energy(self):
        """
        Calculate the energy function to evaluate a candidate. 
        """

        U = -(self.N_ue * self.R_ue - self.N_bs * self.C_bs)

    # start with set of e.g. 25 chosen bs sites?

    def users_within_radius(self, h, k):
        """
        Calculate the euclidean distances to find users within the radius
        self.R of a BS centered on (h,k).
        """

        x = self.user_coords[:, 0]
        y = self.user_coords[:, 1]

        print(x.shape)
        print(y.shape)
        
        distance_squared = np.array( (self.user_coords[:, 0]-h)**2 \
                         + (self.user_coords[:, 1]-k)**2 )

        print(distance_squared)

        

