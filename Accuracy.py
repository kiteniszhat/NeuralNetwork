import numpy as np

class Accuracy:
    def __init__(self, classification_type="categorical"):
        self.classification_type = classification_type

    def calculate(self, y_true, y_pred):
        if self.classification_type == "categorical":
            if len(y_true.shape) == 2:
                y_true = np.argmax(y_true, axis=1)
            y_pred = np.argmax(y_pred, axis=1)
            return np.mean(y_true == y_pred)
        elif self.classification_type == "binary":
            y_pred = (y_pred > 0.5).astype(int)
            return np.mean(y_true == y_pred)
        else:
            raise ValueError("Unknown classification_type")