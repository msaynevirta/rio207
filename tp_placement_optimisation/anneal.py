import numpy as np
import matplotlib.pyplot as plt
from random import choice

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
        self.rng = np.random.default_rng()

        self.user_coords = np.array(user_coords) # UE locations and status (served / not)
        self.bs_sites = np.array(bs_sites) # possible BS sites and status (in use / not)

        self.user_coords_candidate = np.array(user_coords)
        self.bs_sites_candidate = np.array(bs_sites)

        self.user_coords_best = np.array(user_coords)
        self.bs_sites_best = np.array(bs_sites)

        self.B_max = B_max # Max number of deployed BSs
        self.R_ue = R_ue # Individual UE revenue
        self.C_bs = C_bs # BS operational cost

        self.R_cell = R_cell # cell radius

        self.T = 5000 # starting temperature
        self.alpha = 0.995
        self.stopping_temperature = 1e-8
        self.stopping_iter = 100000
        self.iteration = 1

        self.energy_cur = float("Inf")
        self.energy_best = float("Inf")

        self.energy_list = []

    def new_users_in_range(self, bs_idx):
        """
        Computes a boolean np.array with users in range of a certain BS located
        in coords (h,k) and solves who are the new ones (to avoid double
        counting revenue).

        @param in_range  0-indexed BS index value

        @return boolean np.array of new users
        """
        h = self.bs_sites_candidate[bs_idx, 0]
        k = self.bs_sites_candidate[bs_idx, 1]

        in_range = np.array( ( (self.user_coords_candidate[:, 0]-h)**2 \
                             + (self.user_coords_candidate[:, 1]-k)**2 ) \
                             < self.R_cell**2 )

        inv_user_coords = np.logical_not(self.user_coords_candidate[:,2])

        return np.logical_and(in_range, inv_user_coords)

    def add_bs(self, bs_idxs):
        """
        A free BS site is chosen at random and a BS is placed at this location.
        Moves the coordinates of a certain BS site from self.bs_sites to
        self.bs_choices.
        """
        bs_idx = bs_idxs[0]

        if not self.bs_sites_candidate[bs_idx, 2] \
           and np.count_nonzero(self.bs_sites_candidate[:,2]) < self.B_max:
            # mark new users in range of the BS
            self.user_coords_candidate[:,2] =  \
                                    np.where( self.new_users_in_range(bs_idx), \
                                    bs_idx + 1, \
                                    self.user_coords_candidate[:,2] )

            self.bs_sites_candidate[bs_idx, 2] = True # mark bs_site to be in use

    def remove_bs(self, bs_idxs):
        """
        A free BS site is chosen at random and a BS is placed at this location.
        Returns the coordinates of a certain BS site from self.bs_choices to
        self.bs_sites.
        """
        bs_idx = bs_idxs[1]

        if self.bs_sites_candidate[bs_idx, 2]:
            self.user_coords_candidate[:,2] = \
                      np.where( self.user_coords_candidate[:,2] == bs_idx + 1, \
                      0,
                      self.user_coords_candidate[:,2] )
            self.bs_sites_candidate[bs_idx, 2] = False # mark bs_site to be in use

    def move_bs(self, bs_idxs):
        """
        A free BS site is chosen at random and a BS is placed at this location.
        Swaps two sets of coordinates between self.bs_choices and self.bs_sites.
        """
        self.remove_bs(bs_idxs)
        self.add_bs(bs_idxs)

    def candidate_energy(self):
        """
        Calculate the energy function to evaluate a candidate. 
        """
        N_ue = np.count_nonzero(self.user_coords_candidate[:,2])
        N_bs = np.count_nonzero(self.bs_sites_candidate[:,2])

        return -(N_ue * self.R_ue - N_bs * self.C_bs)

    def p_accept(self, energy):
        """
        Probability of accepting if the candidate is worse than current.
        Depends on the current temperature and difference between candidate and current.
        """
        return np.exp(-abs(energy - self.energy_cur) / self.T)

    def new_candidate(self):
        bs_idxs = self.rng.integers(0, high=self.B_max, size=2)
        methods = [self.add_bs, self.remove_bs, self.move_bs]

        choice(methods)(bs_idxs) # run one of the three elemental ops

    def accept_cur(self, energy):
        self.energy_cur = energy
        self.user_coords = self.user_coords_candidate
        self.bs_sites = self.bs_sites_candidate

    def accept_best(self, energy):
        self.energy_cur = energy
        self.user_coords_best = self.user_coords_candidate
        self.bs_sites_best = self.bs_sites_candidate


    def accept(self):
        """
        Accept with probability 1 if candidate is better than current.
        Accept with probabilty p_accept(..) if candidate is worse.
        """
        energy = self.candidate_energy()
        if energy < self.energy_cur:
            self.accept_cur(energy)

            if energy < self.energy_best:
                self.accept_best(energy)
        else:
            if self.rng.random() < self.p_accept(energy):
                self.accept_cur(energy)

    def initial_config(self):
        N_bs = self.rng.integers(0, high=self.B_max)
        for i in range(N_bs):
            self.new_candidate()

        energy = self.candidate_energy()

        self.accept_cur(energy)
        self.accept_cur(energy)

        print("best candidate energy", self.energy_best)

    # start with set of e.g. 15 chosen bs sites?
    def anneal(self):
        """
        Execute simulated annealing algorithm.
        """
        # Initialize with a random amount of BS sites chosen
        self.initial_config()

        print("Starting annealing.")
        while self.T >= self.stopping_temperature and self.iteration < self.stopping_iter:
            self.new_candidate()
            self.accept()
            self.T *= self.alpha
            self.iteration += 1

            self.energy_list.append(self.energy_cur)

        print("Best fitness obtained: ", self.energy_best)
        improvement = 100 * (self.energy_list[0] - self.energy_best) / (self.energy_list[0])
        print(f"Improvement over greedy heuristic: {improvement : .2f}%")

        print(self.bs_sites_best)
        print(np.count_nonzero(self.bs_sites_best[:,2]))
