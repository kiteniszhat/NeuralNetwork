import numpy as np
from LossFunc import CategoricalCrossEntropy


class ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, gradient):
        self.derivative_inputs = gradient.copy()
        self.derivative_inputs[self.inputs <= 0] = 0


class Sigmoid:
    def forward(self, inputs):
        self.output = 1 / (1 + np.exp(-inputs))

    def backward(self, gradient):
        self.derivative_inputs = gradient * (self.output * (1 - self.output))


class Softmax:
    def forward(self, inputs):
        self.inputs = inputs
        exp_val = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp_val / np.sum(exp_val, axis=1, keepdims=True)

    def backward(self, gradient):
        self.derivative_inputs = np.empty_like(gradient)
        for index, (single_output, single_gradient) in enumerate(zip(self.output, gradient)):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            self.derivative_inputs[index] = np.dot(jacobian_matrix, single_gradient)


class SoftmaxCategoricalCrossEntropy:
    def __init__(self):
        self.activation = Softmax()
        self.loss = CategoricalCrossEntropy()

    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.calculate(self.output, y_true)

    def backward(self, gradient, y_true):
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        self.derivative_inputs = gradient.copy()
        self.derivative_inputs[range(len(gradient)), y_true] -= 1
        self.derivative_inputs = self.dinputs / len(gradient)
