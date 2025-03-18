from .core import Loss
import numpy as np

class MSELoss(Loss):
    """Mean Squared Error (MSE) Loss function."""
    
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """
        Computes the forward pass of MSE loss.
        
        Args:
            y_pred (np.ndarray): Predicted values.
            y_true (np.ndarray): Ground truth values.
        
        Returns:
            self: The loss instance with computed loss value.
        """
        self.cache['y_pred'] = y_pred
        self.cache['y_true'] = y_true
        self.value = np.mean(np.square(y_pred - y_true))
        return self
    
    def diff(self) -> np.ndarray:
        """
        Computes the gradient of MSE loss with respect to predictions.
        
        Returns:
            np.ndarray: Gradient of loss.
        """
        return 2 * (self.cache['y_pred'] - self.cache['y_true'])

    def __call__(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """
        Allows the loss instance to be called as a function.
        
        Args:
            y_pred (np.ndarray): Predicted values.
            y_true (np.ndarray): Ground truth values.
        
        Returns:
            np.ndarray: Computed loss value.
        """
        return self.forward(y_pred, y_true)

class NLLLoss(Loss):
    """Negative Log-Likelihood (NLL) Loss function."""
    
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """
        Computes the forward pass of NLL loss.
        
        Args:
            y_pred (np.ndarray): Log-probabilities of predicted values.
            y_true (np.ndarray): One-hot encoded ground truth labels.
        
        Returns:
            self: The loss instance with computed loss value.
        """
        self.cache['y_pred'] = y_pred
        self.cache['y_true'] = y_true
        self.value = -np.mean(np.sum(y_true * y_pred, axis=1))
        return self

    def diff(self) -> np.ndarray:
        """
        Computes the gradient of NLL loss with respect to predictions.
        
        Returns:
            np.ndarray: Gradient of loss.
        """
        return -self.cache['y_true'] / (self.cache['y_pred'].shape[0])

    def __call__(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """
        Allows the loss instance to be called as a function.
        
        Args:
            y_pred (np.ndarray): Log-probabilities of predicted values.
            y_true (np.ndarray): One-hot encoded ground truth labels.
        
        Returns:
            np.ndarray: Computed loss value.
        """
        return self.forward(y_pred, y_true)
