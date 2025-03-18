import numpy as np
from .core import Activation

class Sigmoid(Activation):
    """Sigmoid activation function."""
    
    def __init__(self) -> None:
        super().__init__()
        self.cache['y'] = []  # Store the output for backpropagation

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Computes the sigmoid function."""
        self.cache['x'].append(x)
        y = 1 / (1 + np.exp(-x))
        self.cache['y'].append(y)
        return y

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Computes the gradient of the sigmoid function."""
        y = self.cache['y'].pop()
        return grad * y * (1 - y)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

class Tanh(Activation):
    """Hyperbolic tangent activation function."""
        
    def __init__(self) -> None:
        super().__init__()
        self.cache['y'] = []  # Store the output for backpropagation

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Computes the tanh function."""
        self.cache['x'].append(x)
        y = np.tanh(x)
        self.cache['y'].append(y)
        return y

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Computes the gradient of the tanh function."""
        self.cache['x'].pop()
        return grad * (1 - np.square(self.cache['y'].pop()))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

class ReLU(Activation):
    """Rectified Linear Unit (ReLU) activation function."""
            
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Computes the ReLU function."""
        self.cache['x'].append(x)
        return np.maximum(0, x)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Computes the gradient of the ReLU function."""
        x = self.cache['x'].pop()
        dx = np.ones_like(x)
        dx[x < 0] = 0  # Gradient is 0 for negative inputs
        return grad * dx

class LogSoftmax(Activation):
    """Log-Softmax activation function."""
    
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Computes the log-softmax function."""
        self.cache['x'] = x
        max_vals = np.max(x, axis=1, keepdims=True)  # For numerical stability
        log_sum_exp = np.log(np.sum(np.exp(x - max_vals), axis=1, keepdims=True))
        log_softmax_output = x - max_vals - log_sum_exp
        self.cache['y'] = log_softmax_output
        return log_softmax_output

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Computes the gradient of the log-softmax function."""
        log_softmax_output = self.cache['y']
        softmax_output = np.exp(log_softmax_output)
        return grad - np.sum(grad, axis=1, keepdims=True) * softmax_output

class Softmax(Activation):
    """Softmax activation function."""
    
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Computes the softmax function."""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))  # Numerical stability
        softmax_output = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
        self.cache['y'] = softmax_output
        return softmax_output

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Computes the gradient of the softmax function."""
        softmax_output = self.cache['y']
        batch_size = softmax_output.shape[0]
        grad_input = np.zeros_like(grad)
        
        # Compute Jacobian for each example in the batch
        for i in range(batch_size):
            y = softmax_output[i]
            jacobian = np.diagflat(y) - np.dot(y.reshape(-1, 1), y.reshape(1, -1))
            grad_input[i] = np.dot(grad[i], jacobian)
            
        return grad_input
