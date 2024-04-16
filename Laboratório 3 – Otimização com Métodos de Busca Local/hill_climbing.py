from math import inf


def hill_climbing(cost_function, neighbors, theta0, epsilon, max_iterations):
    """
    Executes the Hill Climbing (HC) algorithm to minimize (optimize) a cost function.

    :param cost_function: function to be minimized.
    :type cost_function: function.
    :param neighbors: function which returns the neighbors of a given point.
    :type neighbors: list of numpy.array.
    :param theta0: initial guess.
    :type theta0: numpy.array.
    :param epsilon: used to stop the optimization if the current cost is less than epsilon.
    :type epsilon: float.
    :param max_iterations: maximum number of iterations.
    :type max_iterations: int.
    :return theta: local minimum.
    :rtype theta: numpy.array.
    :return history: history of points visited by the algorithm.
    :rtype history: list of numpy.array.
    """
    theta = theta0
    history = [theta0]
    iteration = 0

    #j: função custo
    j = {}
    j[None, None] = inf
    j[theta[0], theta[1]] = cost_function(theta)
    # Todo: Implement Hill Climbing

    while j[theta[0], theta[1]] > epsilon and iteration < max_iterations:
        best = [None, None]
        for neighbor in neighbors(theta):
            j[neighbor[0], neighbor[1]] = cost_function(neighbor)
            if j[neighbor[0], neighbor[1]] < j[best[0], best[1]]:
                best = neighbor

        if j[best[0], best[1]] > j[theta[0], theta[1]]:
            return theta, history
        theta = best
        history.append(theta)
        iteration += 1

    return theta, history
