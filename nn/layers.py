import numpy as np
from .core import Layer

class Linear(Layer):
    """Fully connected linear layer for a neural network."""
    
    def __init__(self, input_size: int, output_size: int, init_strategy: str="he") -> None:
        """
        Initializes a linear layer.
        
        Args:
            input_size (int): Number of input features.
            output_size (int): Number of output features.
            init_strategy (str): Strategy for weight initialization (default: "he").
        """
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.weights = {}
        self.init_weights(init_strategy)

    def init_weights(self, strategy: str="he") -> None:
        """
        Initializes layer weights and biases based on the given strategy.
        
        Args:
            strategy (str): Weight initialization strategy. Options: "he", "random", "xavier", "normal".
        """
        if strategy == "he":
            self.weights['w'] = np.random.normal(0, np.sqrt(2.0 / self.input_size), (self.output_size, self.input_size))
            self.weights['b'] = np.random.normal(0, np.sqrt(2.0 / self.input_size), (self.output_size, 1))
        elif strategy == "random":
            self.weights['w'] = np.random.normal(0, 1, (self.output_size, self.input_size))
            self.weights['b'] = np.random.normal(0, 1, (self.output_size, 1))
        elif strategy == "xavier":
            self.weights['w'] = np.random.normal(0, np.sqrt(6.0 / self.input_size), (self.output_size, self.input_size))
            self.weights['b'] = np.random.normal(0, np.sqrt(6.0 / self.input_size), (self.output_size, 1))
        elif strategy == "normal":
            self.weights['w'] = np.random.normal(0, 0.01, (self.output_size, self.input_size))
            self.weights['b'] = np.random.normal(0, 0.01, (self.output_size, 1))
        else:
            raise NotImplementedError("Unsupported initialization strategy")
        
        # Initialize gradients
        self.weights['dw'] = np.zeros_like(self.weights['w'])
        self.weights['db'] = np.zeros_like(self.weights['b'])

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Performs the forward pass of the linear layer.
        
        Args:
            x (np.ndarray): Input data of shape (batch_size, input_size).
        
        Returns:
            np.ndarray: Output data of shape (batch_size, output_size).
        """
        self.cache['x'] = x  # Store input for backpropagation
        return np.dot(x, self.weights['w'].T) + self.weights['b'].T

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """
        Computes gradients for backpropagation.
        
        Args:
            grad (np.ndarray): Gradient of loss with respect to output.
        
        Returns:
            np.ndarray: Gradient of loss with respect to input.
        """
        self.weights['dw'] = np.dot(self.cache['x'].T, grad).T
        self.weights['db'] = np.sum(grad, axis=0, keepdims=True).T
        return np.dot(grad, self.weights['w'])  # Return gradient w.r.t input

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Allows the layer instance to be called as a function.
        
        Args:
            x (np.ndarray): Input data.
        
        Returns:
            np.ndarray: Output data after forward pass.
        """
        return self.forward(x)
