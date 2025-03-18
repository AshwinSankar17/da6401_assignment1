from abc import ABC, abstractmethod
from typing import TypedDict, Dict, List, Tuple

import numpy as np

class AutoDiff(ABC):
    """Abstract base class for automatic differentiation."""

    def __init__(self) -> None:
        self.cache: dict = {}  # Stores intermediate values for backpropagation

    @abstractmethod
    def forward(self, *args) -> None:
        """Performs the forward pass."""
        pass

    @abstractmethod
    def backward(self, *args) -> None:
        """Computes gradients during backpropagation."""
        pass

    @abstractmethod
    def __call__(self, *args) -> None:
        """Allows the class instance to be called as a function."""
        pass

class Weights(TypedDict):
    """Defines the structure for layer weights and gradients."""
    w: np.ndarray  # Weight matrix
    b: np.ndarray  # Bias vector
    dw: np.ndarray  # Gradient of weights
    db: np.ndarray  # Gradient of biases

class Layer(AutoDiff):
    """Abstract base class for neural network layers."""
    weights: Weights = {}

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, *args) -> None:
        """Computes the forward pass of the layer."""
        pass

    @abstractmethod
    def backward(self, *args) -> None:
        """Computes the backward pass of the layer."""
        pass

    @abstractmethod
    def init_weights(self, *args) -> None:
        """Initializes the weights of the layer."""
        pass

class Loss(AutoDiff):
    """Abstract base class for loss functions."""

    def __init__(self, model) -> None:
        super().__init__()
        self.model = model  # Reference to the model
        self.value: np.float32 = 0.0  # Loss value
        self.reg_value: float = 0.0  # Regularization value

    @abstractmethod
    def forward(self, *args) -> None:
        """Computes the loss value."""
        pass

    @abstractmethod
    def diff(self, *args) -> None:
        """Computes the derivative of the loss function."""
        pass

    def backward(self) -> None:
        """Performs backpropagation through the loss function."""
        y_hat = self.diff()
        y_hat = self.model.output_activation.backward(y_hat)
        L = len(self.model.layers)
        for i, layer in enumerate(self.model.layers[::-1]):
            layer.backward(y_hat)
            if L - i - 1 >= 1:
                l__h_prev = np.dot(y_hat, layer.weights['w'])
                y_hat = self.model.activation.backward(l__h_prev)

    def __call__(self, *args) -> None:
        self.forward(*args)
        return self
    
class Activation(AutoDiff):
    """Abstract base class for activation functions."""

    def __init__(self) -> None:
        super().__init__()
        self.cache['x'] = []  # Stores input values for backpropagation

    @abstractmethod
    def forward(self, *args) -> None:
        """Computes the forward pass of the activation function."""
        pass

    @abstractmethod
    def backward(self, *args) -> None:
        """Computes the gradient of the activation function."""
        pass

    def __call__(self, *args) -> None:
        return self.forward(*args)

class Optimizer(ABC):
    """Abstract base class for optimization algorithms."""
    
    def __init__(self) -> None:
        self.params = {}  # Stores parameters for optimization

    @abstractmethod
    def step(self, *args) -> None:
        """Performs an optimization step."""
        pass

    @abstractmethod
    def zero_grad(self, *args) -> None:
        """Resets gradients to zero."""
        pass

class Module(ABC):
    """Abstract base class for neural network models."""
    
    @abstractmethod
    def __init__(self) -> None:
        pass

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Performs the forward pass through the model."""
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.activation(x)
        y = self.layers[-1](x)
        o = self.output_activation(y)
        return o
    
    def __call__(self, x) -> np.ndarray:
        """Allows the model instance to be called as a function."""
        return self.forward(x)
    
    @property
    def parameters(self) -> List[Weights]:
        """Returns a list of model parameters."""
        return [layer.weights for layer in self.layers]
    
    @property
    def state_dict(self) -> Dict[str, np.ndarray]:
        """Returns the model's parameters as a dictionary."""
        state = {}
        for i, layer in enumerate(self.layers):
            state[f'layer_{i}_w'] = layer.weights['w']
            state[f'layer_{i}_b'] = layer.weights['b']
        return state

    def load_state_dict(self, state_dict: Dict[str, np.ndarray]) -> None:
        """Loads model parameters from a dictionary."""
        for i, layer in enumerate(self.layers):
