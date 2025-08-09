import numpy as np


class CategoricalCrossEntropy:
    def forward(self, y_true, y_pred):
        y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)
        loss = -np.sum(y_true * np.log(y_pred_clipped), axis=1)
        return loss, np.mean(loss)

class MeanSquaredError:
    def forward(self, y_true, y_pred):
        loss = (y_true - y_pred)**2
        return loss, np.mean(loss)