import numpy as np
from ActivationFunc import ReLU
from ActivationFunc import Softmax


class DenseLayer:
    def __init__(self, n_inputs, n_neurons, activation=ReLU):
        self.activation = activation()
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        z = np.dot(inputs, self.weights) + self.biases
        self.output = self.activation.forward(z)


# 3 wejścia, 2 neurony, aktywacja ReLU
layer = DenseLayer(3, 2, Softmax)

# przykładowe dane wejściowe
X = np.array([[1, 2, 3], [4, 5, 6]])

layer.forward(X)
print(layer.output)