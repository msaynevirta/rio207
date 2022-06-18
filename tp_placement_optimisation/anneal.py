import numpy as np
import matplotlib.pyplot as plt

class Anneal(object):
    def __init__(self, user_coords, bs_sites, B_max, R_ue, C_bs, R_cell):
        """
        Constructor for the BS / UE simulated anneal class
        
        @param user_coords  np.array of 2D user coordinates
        @param bs_sites     np.array of 2D bs sites
        @param N_ue         total number of users
        @param N_s          number of possible BS sites
        @param B_max        max amount of deployed BSs
        @param R_ue         revenue per UE
        @param C_bs         operating cost per BS
        @param R_cell       cell radius in meters
        """
        self.user_coords = np.array(user_coords) # UE locations and status (served / not)
        self.bs_sites = bs_sites # possible BS sites and status (in use / not)
        self.N_bs = 0 # number of chosen BS sites

        self.B_max = B_max # Max number of deployed BSs
        self.R_ue = R_ue # Individual UE revenue
        self.C_bs = C_bs # BS operational cost

        self.R_cell = R_cell # cell radius

        self.T = 5000 # starting temperature
        self.alpha = 0.995
        self.stopping_temperature = 1e-8
        self.stopping_iter = 100000
        self.iteration = 1

        self.best_solution = None
        self.highest_energy = float("Inf")
        self.energy_list = []

    def new_users_in_range(self, bs_idx):
        """
        Computes a boolean np.array with users in range of a certain BS located
        in coords (h,k) and solves who are the new ones (to avoid double
        counting revenue).

        @param in_range  0-indexed BS index value

        @return boolean np.array of new users
        """
        h = self.bs_sites[bs_idx, 0]
        k = self.bs_sites[bs_idx, 1]

        in_range = np.array( ( (self.user_coords[:, 0]-h)**2 \
                                     + (self.user_coords[:, 1]-k)**2 ) \
                                     < self.R_cell**2 )

        inv_user_coords = np.logical_not(self.user_coords[:,2])

        return np.logical_and(in_range, inv_user_coords)

    def add_bs(self, bs_idx):
        """
        A free BS site is chosen at random and a BS is placed at this location.
        Moves the coordinates of a certain BS site from self.bs_sites to
        self.bs_choices.
        """
        if self.N_bs < self.B_max and not self.bs_sites[bs_idx, 2]:
            # mark new users in range of the BS
            self.user_coords[:,2] = np.where( self.new_users_in_range(bs_idx), \
                                          bs_idx + 1,
                                          self.user_coords[:,2] )

            self.bs_sites[bs_idx, 2] = True # mark bs_site to be in use
            self.N_bs += 1

    def remove_bs(self, bs_idx):
        """
        A free BS site is chosen at random and a BS is placed at this location.
        Returns the coordinates of a certain BS site from self.bs_choices to
        self.bs_sites.
        """
        if self.bs_sites[bs_idx, 2]:
            self.user_coords[:,2] = np.where( self.user_coords[:,2] == bs_idx + 1, \
                                              0,
                                              self.user_coords[:,2] )
            self.bs_sites[bs_idx, 2] = False # mark bs_site to be in use
            self.N_bs -= 1

    def move_bs(self, old_idx, new_idx):
        """
        A free BS site is chosen at random and a BS is placed at this location.
        Swaps two sets of coordinates between self.bs_choices and self.bs_sites.
        """
        self.remove_bs(old_idx)
        self.add_bs(new_idx)

    def energy(self):
        """
        Calculate the energy function to evaluate a candidate. 
        """

        U = -(self.N_ue * self.R_ue - self.N_bs * self.C_bs)

    # start with set of e.g. 25 chosen bs sites?
