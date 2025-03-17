from .core import Loss
import numpy as np

class MSELoss(Loss):
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        self.cache['y_pred'] = y_pred
        self.cache['y_true'] = y_true
        self.value = np.mean(np.square(y_pred - y_true))
        return self
    
    def diff(self) -> np.ndarray:
        return 2 * (self.cache['y_pred'] - self.cache['y_true'])

    def __call__(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        return self.forward(y_pred, y_true)

class NLLLoss(Loss):
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        self.cache['y_pred'] = y_pred
        self.cache['y_true'] = y_true
        self.value = -np.mean(np.sum(y_true * y_pred, axis=1))
        return self

    def diff(self) -> np.ndarray:
        return -self.cache['y_true'] / (self.cache['y_pred'].shape[0])

    def __call__(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        return self.forward(y_pred, y_true)