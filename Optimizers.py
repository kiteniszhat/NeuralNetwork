from DenseLayer import DenseLayer

class StochasticGradientDescent:
    def __init__(self, learning_rate=1., decay=0.):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iteration = 0

    def update_params(self, dense_layer: DenseLayer):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iteration))
            self.iteration += 1
        dense_layer.weights -= self.learning_rate * dense_layer.derivative_weights
        dense_layer.biases -= self.learning_rate * dense_layer.derivative_biases
