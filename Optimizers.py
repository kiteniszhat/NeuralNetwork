import numpy as np

from DenseLayer import DenseLayer


class StochasticGradientDescent:
    def __init__(self, learning_rate=1., decay=0., momentum=0.):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iteration = 0
        self.momentum = momentum

    def before_updating_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iteration))

    def update_params(self, dense_layer: DenseLayer):
        if self.momentum:
            if not hasattr(dense_layer, 'weight_momentums'):
                dense_layer.weight_momentums = np.zeros_like(dense_layer.weights)
                dense_layer.bias_momentums = np.zeros_like(dense_layer.biases)
            
            weight_updates = self.momentum * dense_layer.weight_momentums - self.current_learning_rate * dense_layer.derivative_weights
            dense_layer.weight_momentums = weight_updates
            
            bias_updates = self.momentum * dense_layer.bias_momentums - self.current_learning_rate * dense_layer.derivative_biases
            dense_layer.bias_momentums = bias_updates
            
            dense_layer.weights += weight_updates
            dense_layer.biases += bias_updates
        else:
            dense_layer.weights -= self.current_learning_rate * dense_layer.derivative_weights
            dense_layer.biases -= self.current_learning_rate * dense_layer.derivative_biases

    def after_updating_params(self):
        self.iteration += 1
