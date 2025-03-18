from .core import Optimizer
import numpy as np


class SGDM(Optimizer):
    """
    Stochastic Gradient Descent with Momentum optimizer.
    
    This optimizer implements SGD with momentum, with optional Nesterov acceleration
    and weight decay (L2 regularization).
    
    Args:
        parameters (list): List of parameter dictionaries containing weights 'w',
                          biases 'b', and their gradients 'dw', 'db'.
        learning_rate (float, optional): Step size for parameter updates. Defaults to 1e-3.
        momentum (float, optional): Momentum coefficient for gradient accumulation. Defaults to 0.0.
        weight_decay (float, optional): L2 regularization coefficient. Defaults to 1e-2.
        epsilon (float, optional): Small constant for numerical stability. Defaults to 1e-12.
        nesterov (bool, optional): Whether to use Nesterov accelerated gradient. Defaults to False.
    """
    def __init__(
        self,
        parameters: list,
        learning_rate: float = 1e-3,
        momentum: float = 0.0,
        weight_decay: float = 1e-2,
        epsilon: float = 1e-12,
        nesterov=False,
    ) -> None:
        super().__init__()
        self.parameters = parameters
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.epsilon = epsilon
        self.nesterov = nesterov
        self.step_count = 0
        # Initialize velocity terms for each parameter
        for i, layer in enumerate(self.parameters, start=1):
            self.params[f"vdW{i}"] = np.zeros_like(layer["dw"])
            self.params[f"vdB{i}"] = np.zeros_like(layer["db"])

    def step(self) -> None:
        """
        Performs a single optimization step.
        
        Updates the parameters using momentum (with optional Nesterov acceleration)
        and applies weight decay.
        """
        self.step_count += 1
        for i, layer in enumerate(self.parameters, start=1):
            # Apply weight decay directly to the weights (L2 regularization)
            layer["w"] -= self.weight_decay * layer["w"]
            layer["b"] -= self.weight_decay * layer["b"]
            
            if not self.nesterov:
                # Standard momentum update
                # Update velocity terms
                self.params[f"vdW{i}"] = (
                    self.momentum * self.params[f"vdW{i}"]
                    + self.learning_rate * layer["dw"]
                )
                self.params[f"vdB{i}"] = (
                    self.momentum * self.params[f"vdB{i}"]
                    + self.learning_rate * layer["db"]
                )
                # Update parameters with velocity
                layer["w"] -= self.learning_rate * self.params[f"vdW{i}"]
                layer["b"] -= self.learning_rate * self.params[f"vdB{i}"]
            else:
                # Nesterov accelerated gradient update
                # Store previous velocities
                vdw_prev = self.params[f"vdW{i}"]
                vdb_prev = self.params[f"vdB{i}"]
                # Update velocity terms
                self.params[f"vdW{i}"] = (
                    self.momentum * self.params[f"vdW{i}"]
                    - self.learning_rate * layer["dw"]
                )
                self.params[f"vdB{i}"] = (
                    self.momentum * self.params[f"vdB{i}"]
                    - self.learning_rate * layer["db"]
                )
                # Apply Nesterov update
                layer["w"] += self.learning_rate * (
                    self.momentum * vdw_prev
                    + (1 - self.momentum) * self.params[f"vdW{i}"]
                )
                layer["b"] += self.learning_rate * (
                    self.momentum * vdb_prev
                    + (1 - self.momentum) * self.params[f"vdB{i}"]
                )

    def zero_grad(self, *args) -> None:
        """
        Clears the gradients of all parameters.
        
        Should be called before computing gradients for the next batch.
        """
        for layer in self.parameters:
            layer["dw"] = np.zeros_like(layer["dw"])
            layer["db"] = np.zeros_like(layer["db"])


class RMSProp(Optimizer):
    """
    Root Mean Square Propagation optimizer.
    
    Maintains a moving average of squared gradients to normalize the gradient step.
    Includes weight decay (L2 regularization).
    
    Args:
        parameters (list): List of parameter dictionaries containing weights 'w',
                          biases 'b', and their gradients 'dw', 'db'.
        learning_rate (float, optional): Step size for parameter updates. Defaults to 1e-3.
        alpha (float, optional): Decay rate for squared gradient accumulation. Defaults to 0.9.
        weight_decay (float, optional): L2 regularization coefficient. Defaults to 1e-2.
        epsilon (float, optional): Small constant for numerical stability. Defaults to 1e-12.
    """
    def __init__(
        self,
        parameters: list,
        learning_rate: float = 1e-3,
        alpha: float = 0.9,
        weight_decay: float = 1e-2,
        epsilon: float = 1e-12,
    ) -> None:
        super().__init__()
        self.parameters = parameters
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.weight_decay = weight_decay
        self.epsilon = epsilon
        self.step_count = 0
        # Initialize squared gradient accumulators
        for i, layer in enumerate(self.parameters, start=1):
            self.params[f"sdW{i}"] = np.zeros_like(layer["dw"])
            self.params[f"sdB{i}"] = np.zeros_like(layer["db"])

    def step(self) -> None:
        """
        Performs a single optimization step.
        
        Updates the parameters using RMSProp's adaptive learning rate based on
        the moving average of squared gradients, and applies weight decay.
        """
        self.step_count += 1
        for i, layer in enumerate(self.parameters, start=1):
            # Apply weight decay directly to the weights (L2 regularization)
            layer["w"] -= self.weight_decay * layer["w"]
            layer["b"] -= self.weight_decay * layer["b"]
            
            # Update squared gradient accumulators with exponential moving average
            self.params[f"sdW{i}"] = (
                self.alpha * self.params[f"sdW{i}"]
                + (1 - self.alpha) * layer["dw"] ** 2
            )
            self.params[f"sdB{i}"] = (
                self.alpha * self.params[f"sdB{i}"]
                + (1 - self.alpha) * layer["db"] ** 2
            )
            
            # Update parameters with adaptive learning rate
            layer["w"] -= (
                self.learning_rate
                * layer["dw"]
                / (np.sqrt(self.params[f"sdW{i}"]) + self.epsilon)
            )
            layer["b"] -= (
                self.learning_rate
                * layer["db"]
                / (np.sqrt(self.params[f"sdB{i}"]) + self.epsilon)
            )

    def zero_grad(self, *args) -> None:
        """
        Clears the gradients of all parameters.
        
        Should be called before computing gradients for the next batch.
        """
        for layer in self.parameters:
            layer["dw"] = np.zeros_like(layer["dw"])
            layer["db"] = np.zeros_like(layer["db"])


class AdamW(Optimizer):
    """
    Adaptive Moment Estimation with decoupled weight decay (AdamW) optimizer.
    
    Combines the benefits of RMSProp and momentum, with bias correction
    and decoupled weight decay. Optionally supports Nesterov momentum.
    
    Args:
        parameters (list): List of parameter dictionaries containing weights 'w',
                          biases 'b', and their gradients 'dw', 'db'.
        learning_rate (float, optional): Step size for parameter updates. Defaults to 1e-3.
        beta_1 (float, optional): Exponential decay rate for first moment. Defaults to 0.9.
        beta_2 (float, optional): Exponential decay rate for second moment. Defaults to 0.995.
        weight_decay (float, optional): Decoupled weight decay coefficient. Defaults to 1e-2.
        nesterov (bool, optional): Whether to use Nesterov momentum. Defaults to False.
        epsilon (float, optional): Small constant for numerical stability. Defaults to 1e-8.
    """
    def __init__(
        self,
        parameters: list,
        learning_rate: float = 1e-3,
        beta_1: float = 0.9,
        beta_2: float = 0.995,
        weight_decay: float = 1e-2,
        nesterov=False,
        epsilon: float = 1e-8,
    ) -> None:
        super().__init__()
        self.parameters = parameters
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.epsilon = epsilon
        self.step_count = 0
        # Initialize first and second moment accumulators
        for i, layer in enumerate(self.parameters, start=1):
            self.params[f"vdW{i}"] = np.zeros_like(layer["dw"])  # First moment (momentum)
            self.params[f"vdB{i}"] = np.zeros_like(layer["db"])
            self.params[f"sdW{i}"] = np.zeros_like(layer["dw"])  # Second moment (RMSProp)
            self.params[f"sdB{i}"] = np.zeros_like(layer["db"])

    def step(self) -> None:
        """
        Performs a single optimization step.
        
        Updates the parameters using Adam with bias correction and
        decoupled weight decay. Optionally uses Nesterov momentum.
        """
        self.step_count += 1
        
        # Calculate bias-corrected learning rate
        lr = (
            self.learning_rate
            * np.sqrt(1 - self.beta_2**self.step_count)
            / (1 - self.beta_1**self.step_count + self.epsilon)
        )
        
        for i, layer in enumerate(self.parameters, start=1):
            # Apply weight decay directly to the weights (decoupled from adaptive updates)
            layer["w"] -= self.learning_rate * self.weight_decay * layer["w"]
            layer["b"] -= self.learning_rate * self.weight_decay * layer["b"]
            
            if not self.nesterov:
                # Standard Adam update
                # Update first moment (momentum)
                self.params[f"vdW{i}"] = (
                    self.beta_1 * self.params[f"vdW{i}"]
                    + (1 - self.beta_1) * layer["dw"]
                )
                self.params[f"vdB{i}"] = (
                    self.beta_1 * self.params[f"vdB{i}"]
                    + (1 - self.beta_1) * layer["db"]
                )
                
                # Update second moment (RMSProp)
                self.params[f"sdW{i}"] = self.beta_2 * self.params[f"sdW{i}"] + (
                    1 - self.beta_2
                ) * np.square(layer["dw"])
                self.params[f"sdB{i}"] = self.beta_2 * self.params[f"sdB{i}"] + (
                    1 - self.beta_2
                ) * np.square(layer["db"])
                
                # Update parameters with bias-corrected moments
                layer["w"] -= (
                    lr / (np.sqrt(self.params[f"sdW{i}"] + self.epsilon))
                ) * self.params[f"vdW{i}"]
                layer["b"] -= (
                    lr / (np.sqrt(self.params[f"sdB{i}"] + self.epsilon))
                ) * self.params[f"vdB{i}"]
            else:
                # Nesterov Adam update
                # Update first moment (momentum)
                self.params[f"vdW{i}"] = (
                    self.beta_1 * self.params[f"vdW{i}"]
                    + (1 - self.beta_1) * layer["dw"]
                )
                self.params[f"vdB{i}"] = (
                    self.beta_1 * self.params[f"vdB{i}"]
                    + (1 - self.beta_1) * layer["db"]
                )
                
                # Update second moment (RMSProp)
                self.params[f"sdW{i}"] = self.beta_2 * self.params[f"sdW{i}"] + (
                    1 - self.beta_2
                ) * np.square(layer["dw"])
                self.params[f"sdB{i}"] = self.beta_2 * self.params[f"sdB{i}"] + (
                    1 - self.beta_2
                ) * np.square(layer["db"])
                
                # Apply Nesterov update with bias correction
                layer["w"] -= (
                    lr
                    * (
                        self.beta_1 * self.params[f"vdW{i}"]
                        + ((1 - self.beta_1) * layer["dw"])
                        / (1 - self.beta_1**self.step_count)
                    )
                    / (np.sqrt(self.params[f"sdW{i}"]) + self.epsilon)
                )
                layer["b"] -= (
                    lr
                    * (
                        self.beta_1 * self.params[f"vdB{i}"]
                        + ((1 - self.beta_1) * layer["db"])
                        / (1 - self.beta_1**self.step_count)
                    )
                    / (np.sqrt(self.params[f"sdB{i}"]) + self.epsilon)
                )

    def zero_grad(self, *args) -> None:
        """
        Clears the gradients of all parameters.
        
        Should be called before computing gradients for the next batch.
        """
        for layer in self.parameters:
            layer["dw"] = np.zeros_like(layer["dw"])
            layer["db"] = np.zeros_like(layer["db"])
