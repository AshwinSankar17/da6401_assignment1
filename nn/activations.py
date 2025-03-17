import numpy as np
from .core import Activation

class Sigmoid(Activation):
    
    def __init__(self) -> None:
        super().__init__()
        self.cache['y'] = []

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.cache['x'].append(x)
        y = 1 / (1 + np.exp(-x))
        self.cache['y'].append(y)
        return y

    def backward(self, grad: np.ndarray) -> np.ndarray:
        y = self.cache['y'].pop()
        return grad * y * (1 - y)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

class Tanh(Activation):
        
    def __init__(self) -> None:
        super().__init__()
        self.cache['y'] = []

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.cache['x'].append(x)
        y = np.tanh(x)
        self.cache['y'].append(y)
        return y

    def backward(self, grad: np.ndarray) -> np.ndarray:
        self.cache['x'].pop()
        return grad * (1 - np.square(self.cache['y'].pop()))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

class ReLU(Activation):
            
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.cache['x'].append(x)
        y = np.maximum(0, x)
        return y

    def backward(self, grad: np.ndarray) -> np.ndarray:
        x = self.cache['x'].pop()
        dx = np.ones_like(x)
        dx[x < 0] = 0
        return grad * dx

class LogSoftmax(Activation):
    
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.cache['x'] = x
        max_vals = np.max(x, axis=1, keepdims=True)
        log_sum_exp = np.log(np.sum(np.exp(x - max_vals), axis=1, keepdims=True))
        log_softmax_output = x - max_vals - log_sum_exp
        self.cache['y'] = log_softmax_output
        return log_softmax_output

    def backward(self, grad: np.ndarray) -> np.ndarray:
        log_softmax_output = self.cache['y']
        softmax_output = np.exp(log_softmax_output)
        grad_input = grad - np.sum(grad, axis=1, keepdims=True) * softmax_output
        return grad_input

class Softmax(Activation):
    
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        softmax_output = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
        self.cache['y'] = softmax_output
        return softmax_output

    def backward(self, grad: np.ndarray) -> np.ndarray:
        softmax_output = self.cache['y']
        batch_size = softmax_output.shape[0]
        grad_input = np.zeros_like(grad)
        
        for i in range(batch_size):
            y = softmax_output[i]
            jacobian = np.diagflat(y) - np.dot(y.reshape(-1, 1), y.reshape(1, -1))
            grad_input[i] = np.dot(grad[i], jacobian)
            
        return grad_input