import numpy as np

from .core import Layer

class Linear(Layer):
    def __init__(self, input_size: int, output_size: int, init_strategy: str="he") -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.weights = {}
        self.init_weights(init_strategy)

    def init_weights(self, strategy:str="he") -> None:
        if strategy == "he":
            self.weights['w'] = np.random.normal(0, np.sqrt(2.0 / self.input_size), (self.output_size, self.input_size))
            self.weights['b'] = np.random.normal(0, np.sqrt(2.0 / self.input_size), (self.output_size, 1))
            self.weights['dw'] = np.zeros_like(self.weights['w'])
            self.weights['db'] = np.zeros_like(self.weights['b'])
        elif strategy == "random":
            self.weights['w'] = np.random.normal(0, 1, (self.output_size, self.input_size))
            self.weights['b'] = np.random.normal(0, 1, (self.output_size, 1))
            self.weights['dw'] = np.zeros_like(self.weights['w'])
            self.weights['db'] = np.zeros_like(self.weights['b'])
        elif strategy == "xavier":
            self.weights['w'] = np.random.normal(0, np.sqrt(6.0 / self.input_size), (self.output_size, self.input_size))
            self.weights['b'] = np.random.normal(0, np.sqrt(6.0 / self.input_size), (self.output_size, 1))
            self.weights['dw'] = np.zeros_like(self.weights['w'])
            self.weights['db'] = np.zeros_like(self.weights['b'])
        elif strategy == "normal":
            self.weights['w'] = np.random.normal(0, 0.01, (self.output_size, self.input_size))
            self.weights['b'] = np.random.normal(0, 0.01, (self.output_size, 1))
            self.weights['dw'] = np.zeros_like(self.weights['w'])
            self.weights['db'] = np.zeros_like(self.weights['b'])
        else:
            raise NotImplementedError

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.cache['x'] = x
        return np.dot(x, self.weights['w'].T) + self.weights['b'].T

    def backward(self, grad: np.ndarray) -> np.ndarray:
        
        self.weights['dw'] = np.dot(self.cache['x'].T, grad).T
        self.weights['db'] = np.sum(grad, axis=0, keepdims=True).T

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)