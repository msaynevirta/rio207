import numpy as np
from random import choice

class Anneal(object):
    def __init__(self, user_coords, bs_sites, opt_params, ue_bs_params):
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

        self.user_coords_candidate = np.copy(user_coords)
        self.bs_sites_candidate = np.copy(bs_sites)

        self.user_coords_best = np.copy(user_coords)
        self.bs_sites_best = np.copy(bs_sites)

        self.ue_emf = np.full(user_coords.shape[0], 0, dtype=np.float64)
        self.ue_emf_candidate = np.full(user_coords.shape[0], 0, dtype=np.float64)
        self.ue_emf_best = np.full(user_coords.shape[0], 0, dtype=np.float64)

        self.opt_params = opt_params
        self.ue_bs_params = ue_bs_params

        self.T = 5000 # starting temperature
        self.alpha = 0.997
        self.stopping_temperature = 1e-8
        self.stopping_iter = 100000
        self.iteration = 1

        self.energy_cur = float("Inf")
        self.energy_best = float("Inf")

        self.energy_list = []

    def all_users_in_range(self, bs_idx):
        h = self.bs_sites[bs_idx, 0]
        k = self.bs_sites[bs_idx, 1]
        y_dist_sq = (self.ue_bs_params['h_bs'] - self.ue_bs_params['h_ue'])**2

        return np.array( ( (self.user_coords[:, 0]-h)**2 \
                             + (self.user_coords[:, 1]-k)**2 \
                             + y_dist_sq ) \
                             < self.ue_bs_params['R_cell']**2 )

    def new_users_in_range(self, bs_idx):
        """
        Computes a boolean np.array with users in range of a certain BS located
        in coords (h,k) and solves who are the new ones (to avoid double
        counting revenue).

        @param in_range  0-indexed BS index value

        @return boolean np.array of new users
        """
        in_range = self.all_users_in_range(bs_idx)

        inv_user_coords = np.logical_not(self.user_coords[:,2])

        return np.logical_and(in_range, inv_user_coords)

    def add_bs(self, bs_idxs):
        """
        A free BS site is chosen at random and a BS is placed at this location.
        Moves the coordinates of a certain BS site from self.bs_sites to
        self.bs_choices.
        """
        bs_idx = bs_idxs[0]

        if not self.bs_sites[bs_idx, 2] \
           and np.count_nonzero(self.bs_sites[:,2]) < self.opt_params['B_max']:
            # mark new users in range of the BS
            self.user_coords_candidate[:,2] =  \
                                    np.where( self.new_users_in_range(bs_idx), \
                                    bs_idx + 1, \
                                    self.user_coords[:,2] )

            self.bs_sites_candidate[bs_idx, 2] = True # mark bs_site to be in use
            self.candidate_emf_exposure(bs_idx, 1)

    def remove_bs(self, bs_idxs):
        """
        A free BS site is chosen at random and a BS is placed at this location.
        Returns the coordinates of a certain BS site from self.bs_choices to
        self.bs_sites.
        """
        bs_idx = bs_idxs[1]

        if self.bs_sites[bs_idx, 2]:
            self.user_coords_candidate[:,2] = \
                      np.where( self.user_coords[:,2] == bs_idx + 1, \
                      0,
                      self.user_coords[:,2] )

            self.bs_sites_candidate[bs_idx, 2] = False # mark bs_site to be in use
            self.candidate_emf_exposure(bs_idx, -1)

    def move_bs(self, bs_idxs):
        """
        A free BS site is chosen at random and a BS is placed at this location.
        Swaps two sets of coordinates between self.bs_choices and self.bs_sites.
        """
        self.remove_bs(bs_idxs)
        self.add_bs(bs_idxs)

    def candidate_revenue(self):
        """
        Computes the total revenue produced by certain candidate.
        """
        N_ue = np.count_nonzero(self.user_coords_candidate[:,2])
        N_bs = np.count_nonzero(self.bs_sites_candidate[:,2])

        return (N_ue * self.opt_params['R_ue'] - N_bs * self.opt_params['C_bs'])

    def candidate_emf_exposure(self, bs_idx, sign):
        """
        Computes the total EMF exposure induced by the candidate.
        """
        h = self.bs_sites[bs_idx, 0]
        k = self.bs_sites[bs_idx, 1]

        num = (30 * self.ue_bs_params['P_tx_bs'] * self.ue_bs_params['G_ant_bs'])
        y_dist_sq = (self.ue_bs_params['h_bs'] - self.ue_bs_params['h_ue'])**2
        
        self.ue_emf_candidate = self.ue_emf + num / ( (self.user_coords[:, 0]-h)**2 \
                                             + (self.user_coords[:, 1]-k)**2 \
                                             + y_dist_sq ) \
                                       * sign

    def candidate_energy(self):
        """
        Calculate the energy function to evaluate a candidate. 
        """
        emf_exposure = self.opt_params['include_emf_exposure'] \
                     * np.max(self.ue_emf_candidate) **3

        return -self.candidate_revenue() + emf_exposure

    def p_accept(self, energy):
        """
        Probability of accepting if the candidate is worse than current.
        Depends on the current temperature and difference between candidate and current.
        """
        return np.exp(-abs(energy - self.energy_cur) / self.T)

    def new_candidate(self):
        bs_idxs = self.rng.integers(0, high=self.bs_sites.shape[0], size=2)
        methods = [self.add_bs, self.remove_bs, self.move_bs]

        choice(methods)(bs_idxs) # run one of the three elemental ops

    def accept_cur(self, energy):
        self.energy_cur = energy
        self.user_coords = np.copy(self.user_coords_candidate)
        self.bs_sites = np.copy(self.bs_sites_candidate)
        self.ue_emf = self.ue_emf_candidate

    def accept_best(self, energy):
        self.energy_best = energy
        self.user_coords_best = np.copy(self.user_coords_candidate)
        self.bs_sites_best = np.copy(self.bs_sites_candidate)
        self.ue_emf_best = self.ue_emf_candidate

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
        N_bs = self.rng.integers(low=1, high=self.opt_params['B_max'])
        for _ in range(N_bs):
            self.new_candidate()

        energy = self.candidate_energy()

        self.accept_cur(energy)

    def anneal(self):
        """
        Execute the simulated annealing algorithm.
        """
        print("Starting annealing\n---------------")

        self.initial_config()
        print("BS sites, start =", np.count_nonzero(self.bs_sites[:,2]))

        while self.T >= self.stopping_temperature \
              and self.iteration < self.stopping_iter:
            self.new_candidate()
            self.accept()
            self.T *= self.alpha
            self.iteration += 1

            users = int(np.count_nonzero(self.user_coords_best[:,2]))
            bs = int(np.count_nonzero(self.bs_sites_best[:,2]))

            self.energy_list.append([self.energy_cur, self.energy_best])

        print("\nFinished annealing\n---------------")
        print("Best energy obtained: ", self.energy_best)

        print("EMF over 6 V/m:", np.count_nonzero(self.ue_emf_best > 6.0), \
              "-- EMF over 0.6 V/m:", np.count_nonzero(self.ue_emf_best > 0.6))

        print("Highest EMF exposure [V/m]:", np.max(self.ue_emf_best))

        print("Users served:", np.count_nonzero(self.user_coords_best[:,2]), \
              "-- bs sites:", np.count_nonzero(self.bs_sites_best[:,2]), \
              "-- energy:", self.energy_best)

    def plot_energy(self, ax):
        """
        Plot the current and best energy through iterations.
        """
        ax.set_title('Evolution of energy')
        ax.plot([i for i in range(len(self.energy_list))], self.energy_list)

    def plot_scatter_bs_ue(self, ax):
        ue = np.copy(self.user_coords_best)
        mask = (ue[:, 2] != 0)
        ue = ue[mask, :]

        bs = np.copy(self.bs_sites_best)
        mask = (bs[:, 2] != 0)
        bs = bs[mask, :]

        ax.set_title('BS sites and served UEs')
        ax.scatter(ue[:,0], ue[:,1], c=ue[:,2])
        ax.scatter(bs[:,0], bs[:,1], c='black', s=200, marker="1")
