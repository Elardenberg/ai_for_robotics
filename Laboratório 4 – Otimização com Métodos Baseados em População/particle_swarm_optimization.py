import numpy as np
import random
from math import inf


class Particle:
    """
    Represents a particle of the Particle Swarm Optimization algorithm.
    """
    def __init__(self, lower_bound, upper_bound):
        """
        Creates a particle of the Particle Swarm Optimization algorithm.

        :param lower_bound: lower bound of the particle position.
        :type lower_bound: numpy array.
        :param upper_bound: upper bound of the particle position.
        :type upper_bound: numpy array.
        """
        # Todo: implement
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.x = np.random.uniform(self.lower_bound, self.upper_bound)
        self.delta = self.upper_bound - self.lower_bound
        self.v = np.random.uniform(-self.delta, self.delta)
        self.value = -inf
        self.best_position = self.x
        self.best_value = self.value


class ParticleSwarmOptimization:
    """
    Represents the Particle Swarm Optimization algorithm.
    Hyperparameters:
        inertia_weight: inertia weight.
        cognitive_parameter: cognitive parameter.
        social_parameter: social parameter.

    :param hyperparams: hyperparameters used by Particle Swarm Optimization.
    :type hyperparams: Params.
    :param lower_bound: lower bound of particle position.
    :type lower_bound: numpy array.
    :param upper_bound: upper bound of particle position.
    :type upper_bound: numpy array.
    """
    def __init__(self, hyperparams, lower_bound, upper_bound):
        # Todo: implement
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.hyperparams = hyperparams
        self.particles = []
        self.position_marker = 0

        for i in range(self.hyperparams.num_particles):
            self.particles.append(Particle(lower_bound, upper_bound))

        self.best_global = None
        self.best_global_value = -inf

    def get_best_position(self):
        """
        Obtains the best position so far found by the algorithm.

        :return: the best position.
        :rtype: numpy array.
        """
        # Todo: implement
        return self.best_global

    def get_best_value(self):
        """
        Obtains the value of the best position so far found by the algorithm.

        :return: value of the best position.
        :rtype: float.
        """
        # Todo: implement
        return self.best_global_value

    def get_position_to_evaluate(self):
        """
        Obtains a new position to evaluate.

        :return: position to evaluate.
        :rtype: numpy array.
        """

        return self.particles[self.position_marker].x

    def advance_generation(self):
        """
        Advances the generation of particles. Auxiliary method to be used by notify_evaluation().
        """
        self.rp = np.random.uniform(0.0, 1.0)
        self.rg = np.random.uniform(0.0, 1.0)
        self.particles[self.position_marker].v = (self.hyperparams.inertia_weight * self.particles[self.position_marker].v +
            self.hyperparams.cognitive_parameter * self.rp * (self.particles[self.position_marker].best_position - self.particles[self.position_marker].x) +
            self.hyperparams.social_parameter * self.rg * (self.best_global - self.particles[self.position_marker].x))

        self.particles[self.position_marker].x = self.particles[self.position_marker].x + self.particles[self.position_marker].v

    def notify_evaluation(self, value):
        """
        Notifies the algorithm that a particle position evaluation was completed.

        :param value: quality of the particle position.
        :type value: float.
        """
        # Todo: implement
        self.particles[self.position_marker].value = value
        if self.particles[self.position_marker].value > self.particles[self.position_marker].best_value:
            self.particles[self.position_marker].best_value = self.particles[self.position_marker].value
            self.particles[self.position_marker].best_position = self.particles[self.position_marker].x

        if self.particles[self.position_marker].value > self.best_global_value:
            self.best_global_value = self.particles[self.position_marker].value
            self.best_global = self.particles[self.position_marker].x

        self.advance_generation()

        self.position_marker = (self.position_marker + 1) % self.hyperparams.num_particles
