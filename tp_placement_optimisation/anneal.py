import numpy as np
import matplotlib.pyplot as plt

class Anneal(object):
    def __init__(self, user_coords, bs_sites):
        self.user_coords = user_coords # UE locations
        self.bs_sites = bs_sites # possible BS sites
        self.bs_choices = np.empty() # chosen BS sites
        self.N_bs = 0 # number of chosen BS sites

        self.T = 5000 # starting temperature
        self.alpha = 0.995
        self.stopping_temperature = 1e-8
        self.stopping_iter = 100000
        self.iteration = 1

        self.best_solution = None
        self.best_fitness = float("Inf")
        self.fitness_list = []

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

    def energy():
        pass

    # start with set of e.g. 25 chosen bs sites?