import numpy as np


class CategoricalCrossEntropy:
    def calculate(self, y_true, y_pred):
        return np.mean(self.forward(y_true, y_pred))

    def forward(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred
        y_pred_clipped = np.clip(y_pred, 1e-10, 1 - 1e-10)
        print(y_pred)
        print(y_pred_clipped)
        loss = -np.sum(y_true * np.log(y_pred_clipped), axis=1)
        return loss


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
