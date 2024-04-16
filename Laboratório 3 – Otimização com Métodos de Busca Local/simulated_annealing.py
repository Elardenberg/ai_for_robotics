from math import exp
import random


def simulated_annealing(cost_function, random_neighbor, schedule, theta0, epsilon, max_iterations):
    """
    Executes the Simulated Annealing (SA) algorithm to minimize (optimize) a cost function.

    :param cost_function: function to be minimized.
    :type cost_function: function.
    :param random_neighbor: function which returns a random neighbor of a given point.
    :type random_neighbor: numpy.array.
    :param schedule: function which computes the temperature schedule.
    :type schedule: function.
    :param theta0: initial guess.
    :type theta0: numpy.array.
    :param epsilon: used to stop the optimization if the current cost is less than epsilon.
    :type epsilon: float.
    :param max_iterations: maximum number of iterations.
    :type max_iterations: int.
    :return theta: local minimum.
    :rtype theta: np.array.
    :return history: history of points visited by the algorithm.
    :rtype history: list of np.array.
    """
    theta = theta0
    history = [theta0]
    # Todo: Implement Simulated Annealing
    iteration = 0

    #j: função custo
    j = {}
    j[theta[0], theta[1]] = cost_function(theta)
    # Todo: Implement Hill Climbing

    while j[theta[0], theta[1]] > epsilon and iteration < max_iterations:
        T = schedule(iteration)

        if T < 0:
            return theta, history

        neighbor = random_neighbor(theta)
        j[neighbor[0], neighbor[1]] = cost_function(neighbor)
        deltaE = j[theta[0], theta[1]] - j[neighbor[0], neighbor[1]]

        if deltaE > 0:
            theta = neighbor
        else:
            r = random.uniform(0.0, 1.0)

            if r <= exp(deltaE/T):
                theta = neighbor

        history.append(theta)
        iteration += 1

    return theta, history
