import numpy as np


class DenseLayer:
    def __init__(self, n_inputs, n_neurons):
        self.output = None
        self.inputs = None
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, gradient):
        self.derivative_weights = np.dot(self.inputs.T, gradient)
        self.derivative_biases = np.sum(gradient, axis=0, keepdims=True)
        self.derivative_inputs = np.dot(gradient, self.weights.T)
