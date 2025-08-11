import numpy as np
from ActivationFunc import ReLU

class DenseLayer:
    def __init__(self, n_inputs, n_neurons, activation=ReLU):
        self.activation = activation()
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        z = np.dot(inputs, self.weights) + self.biases
        self.output = self.activation.calculate(z)
