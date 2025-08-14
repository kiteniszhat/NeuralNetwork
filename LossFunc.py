import numpy as np


class CategoricalCrossEntropy:
    def calculate(self, y_true, y_pred):
        return np.mean(self.forward(y_true, y_pred))

    def forward(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred
        y_pred_clipped = np.clip(y_pred, 1e-10, 1 - 1e-10)
        if len(y_true.shape) == 2:
            loss = np.sum(y_true * y_pred_clipped, axis=1)
        elif len(y_true.shape) == 1:
            loss = y_pred_clipped[range(len(y_pred_clipped)), y_true]
        return -np.log(loss)

    # def backward(self, gradient, y_true):


class MeanSquaredError:
    def calculate(self, y_true, y_pred):
        return np.mean(self.forward(y_true, y_pred))

    def forward(self, y_true, y_pred):
        loss = (y_true - y_pred) ** 2
        return loss

    def backward(self, y_true, y_pred):
        outputs = y_pred.shape[1]
        self.derivative_inputs = -2 * (y_true - y_pred) / outputs
        self.derivative_inputs = self.derivative_inputs / len(y_pred)
