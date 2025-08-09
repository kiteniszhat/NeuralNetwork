import numpy as np

class ReLU:
    def forward(self, inputs):
        return np.maximum(0, inputs)


class Sigmoid:
    def forward(self, inputs):
        return 1 / (1 + np.exp(-inputs))


class Softmax:
    def forward(self, inputs):
        exp_val = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        return exp_val / np.sum(exp_val, axis=1, keepdims=True)
