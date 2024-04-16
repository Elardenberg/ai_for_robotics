import numpy as np
from utils import sigmoid, sigmoid_derivative


class NeuralNetwork:
    """
    Represents a two-layers Neural Network (NN) for multi-class classification.
    The sigmoid activation function is used for all neurons.
    """
    def __init__(self, num_inputs, num_hiddens, num_outputs, alpha):
        """
        Constructs a three-layers Neural Network.

        :param num_inputs: number of inputs of the NN.
        :type num_inputs: int.
        :param num_hiddens: number of neurons in the hidden layer.
        :type num_hiddens: int.
        :param num_outputs: number of outputs of the NN.
        :type num_outputs: int.
        :param alpha: learning rate.
        :type alpha: float.
        """
        self.num_inputs = num_inputs
        self.num_hiddens = num_hiddens
        self.num_outputs = num_outputs
        self.alpha = alpha
        self.weights = [None] * 3
        self.biases = [None] * 3
        self.weights[1] = 0.001 * np.random.randn(num_hiddens, num_inputs)
        self.weights[2] = 0.001 * np.random.randn(num_outputs, num_hiddens)
        self.biases[1] = np.zeros((num_hiddens, 1))
        self.biases[2] = np.zeros((num_outputs, 1))

    def forward_propagation(self, inputs):
        """
        Executes forward propagation.
        Notice that the z and a of the first layer (l = 0) are equal to the NN's input.

        :param inputs: inputs to the network.
        :type inputs: (num_inputs, num_samples) numpy array.
        :return z: values computed by applying weights and biases at each layer of the NN.
        :rtype z: 3-dimensional list of (num_neurons[l], num_samples) numpy matrices.
        :return a: activations computed by applying the activation function to z at each layer.
        :rtype a: 3-dimensional list of (num_neurons[l], num_samples) numpy matrices.
        """
        z = [None] * 3
        a = [None] * 3
        z[0] = inputs
        a[0] = inputs
        # Add logic for neural network inference
        # Start coding implementation
        z[1] = self.weights[1]@a[0] + self.biases[1]
        a[1] = sigmoid(z[1])

        z[2] = self.weights[2]@a[1] + self.biases[2]
        a[2] = sigmoid(z[2])
        return z, a

    def compute_cost(self, inputs, expected_outputs):
        """
        Computes the logistic regression cost of this network.

        :param inputs: inputs to the network.
        :type inputs: (num_inputs, num_samples) numpy array.
        :param expected_outputs: expected outputs of the network.
        :type expected_outputs: list of numpy matrices.
        :return: logistic regression cost.
        :rtype: float.
        """
        z, a = self.forward_propagation(inputs)
        y = expected_outputs
        y_hat = a[-1]
        cost = np.mean(-(y * np.log(y_hat) + (1.0 - y) * np.log(1.0 - y_hat)))
        return cost

    def compute_gradient_back_propagation(self, inputs, expected_outputs):
        """
        Computes the gradient with respect to the NN's parameters using back propagation.

        :param inputs: inputs to the network.
        :type inputs: (num_inputs, num_samples) numpy array.
        :param expected_outputs: expected outputs of the network.
        :type expected_outputs: (num_outputs, num_samples) numpy array.
        :return weights_gradient: gradients of the weights at each layer.
        :rtype weights_gradient: 3-dimensional list of numpy arrays.
        :return biases_gradient: gradients of the biases at each layer.
        :rtype biases_gradient: 3-dimensional list of numpy arrays.
        """
        weights_gradient = [None] * 3
        biases_gradient = [None] * 3

        z, a = self.forward_propagation(inputs)

        """
        Parameters dimensions for test_neural_network:

        a[0]: 2 x 200
        a[1]: 10 x 200
        a[2]: 1 x 200
        self.weights[1]: 10 x 2
        self.weights[2]: 1 x 10
        self.biases[1]: 10 x 1
        self.biases[2]: 1 x 1
        expected_outputs: 200,
        z = a
        delta1: 10 x 200
        delta2 = z[2]
        weights_gradient[1] = 2 x 10 x 200
        weights_gradient[2] = 10 x 200

        """

        delta2 = a[2] - expected_outputs
        delta1 = sigmoid_derivative(z[1])*(self.weights[2].T@delta2)

        weights_gradient[2] = [None] * self.num_outputs
        for i in range(self.num_outputs):
            weights_gradient[2][i] = delta2[i]*a[1]

        # weights_gradient[2] = delta2*a[1]
        biases_gradient[2] = delta2
        weights_gradient[1] = [None] * self.num_inputs
        biases_gradient[1] = delta1

        # Add logic to compute the gradients
        for k in range(self.num_inputs):
            weights_gradient[1][k] = delta1*a[0][k]

        return weights_gradient, biases_gradient

    def back_propagation(self, inputs, expected_outputs):
        """
        Executes the back propagation algorithm to update the NN's parameters.

        :param inputs: inputs to the network.
        :type inputs: (num_inputs, num_samples) numpy array.
        :param expected_outputs: expected outputs of the network.
        :type expected_outputs: (num_outputs, num_samples) numpy array.
        """
        weights_gradient, biases_gradient = self.compute_gradient_back_propagation(inputs, expected_outputs)
        # Add logic to update the weights and biases

        for k in range(self.num_hiddens):
            for j in range(self.num_inputs):
                self.weights[1].T[j][k] = self.weights[1].T[j][k] - self.alpha * np.mean(weights_gradient[1][j][k])
            for i in range(self.num_outputs):
                self.weights[2][i][k] = self.weights[2][i][k] - self.alpha * np.mean(weights_gradient[2][i][k])
            self.biases[1][k] = self.biases[1][k] - self.alpha * np.mean(biases_gradient[1][k])

        self.biases[2] = self.biases[2] - self.alpha * np.mean(biases_gradient[2])
