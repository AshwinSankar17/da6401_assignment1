from nn.core import Module
from nn.layers import Linear
from nn.activations import ReLU, Sigmoid, Tanh, LogSoftmax, Softmax
from nn.losses import MSELoss, NLLLoss
from nn.optim import SGDM, AdamW, RMSProp
from nn.utils import DataLoader, Dataset

from dataclasses import dataclass
from tqdm import tqdm

import json
import wandb
import utils
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from argparse import ArgumentParser


@dataclass
class HyperParameters:
    hidden_activations: str = "tanh"
    init_strategy: str = "normal"
    loss_fn: str = "nll"
    optimizer: str = "nadam"
    learning_rate: float = 1e-4
    n_epochs: int = 20
    batch_size: int = 64
    weight_decay: float = 1e-2
    beta_1: float = 0.9
    beta_2: float = 0.995
    hidden_sizes: list = (128, 64, 32)
    epsilon: float = 1e-8


class NeuralNetwork(Module):
    """
    A simple feedforward neural network with configurable hidden layers and activation functions.

    Attributes:
        output_activation (LogSoftmax): The output activation function (log softmax for classification tasks).
        hyperparameters (HyperParameters): Object containing network hyperparameters.
        layers (list): List of fully connected (linear) layers.
        activation (function): Activation function used for hidden layers (ReLU by default).
    """
    def __init__(
        self, input_size: int, output_size: int, hyperparameters: HyperParameters
    ) -> None:
        """
        Initializes the neural network with the given input size, output size, and hyperparameters.

        Args:
            input_size (int): The number of input features.
            output_size (int): The number of output classes/units.
            hyperparameters (HyperParameters): Object containing hyperparameters such as hidden layer sizes,
                                              activation functions, and initialization strategy.
        """
        super().__init__()
        self.output_activation = LogSoftmax() # Apply LogSoftmax at the output layer for classification
        self.hyperparameters = hyperparameters
        self.layers = [] # List to store the linear layers

        # Create the input layer
        fc_in = Linear(
            input_size, hyperparameters.hidden_sizes[0], hyperparameters.init_strategy
        )

        # Select activation function based on user-specified hyperparameters. Default to ReLU if not specified
        self.activation = {
            "relu": ReLU(),
            "sigmoid": Sigmoid(),
            "tanh": Tanh(),
        }.get(hyperparameters.hidden_activations, ReLU())

        self.layers.append(fc_in) # Add input layer to the network

        # Create hidden layers
        for i in range(1, len(hyperparameters.hidden_sizes)):
            fc = Linear(
                hyperparameters.hidden_sizes[i - 1],
                hyperparameters.hidden_sizes[i],
                hyperparameters.init_strategy,
            )
            self.layers.append(fc)

        # Create the output layer
        fc_out = Linear(
            hyperparameters.hidden_sizes[-1], output_size, hyperparameters.init_strategy
        )
        self.layers.append(fc_out)

    def forward(self, x):
        # Pass through hidden layers with activation functions
        for layer in self.layers[:-1]:
            x = layer(x) # Apply linear transformation
            x = self.activation(x) # Apply activation function

        # Apply the final linear transformation and output activation
        x = self.layers[-1](x)
        return self.output_activation(x)


def train(model, train_loader, val_loader, optimizer, loss_fn, n_epochs):
    """
    Trains the given model using the provided data loaders, optimizer, and loss function.
    
    Args:
        model: The neural network model to be trained.
        train_loader: DataLoader for the training dataset.
        val_loader: DataLoader for the validation dataset.
        optimizer: The optimizer used for updating model parameters.
        loss_fn: The loss function used for computing loss.
        n_epochs: Number of training epochs.

    Returns:
        The trained model.
    """
    best_val_loss = float("inf")  # Initialize best validation loss to infinity
    best_val_accuracy = 0  # Initialize best validation accuracy to 0
    
    for epoch in range(n_epochs):
        train_loss = 0.0  # Accumulate training loss
        train_accuracy = 0.0  # Accumulate training accuracy
        
        # Training loop
        for step_idx, (x, y) in tqdm(
            enumerate(train_loader), desc="Training", total=len(train_loader)
        ):
            optimizer.zero_grad()  # Reset gradients
            y_hat = model(x)  # Forward pass
            loss = loss_fn(y_hat, y)  # Compute loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update model parameters
            
            train_loss += loss.value  # Accumulate batch loss
            train_accuracy += (y_hat.argmax(axis=1) == y.argmax(axis=1)).mean()  # Compute batch accuracy
            
            # Log training metrics
            wandb.log(
                {
                    "train/loss": loss.value,
                    "train/acc": (y_hat.argmax(axis=1) == y.argmax(axis=1)).mean(),
                }
            )
        
        # Compute average training loss and accuracy
        train_loss /= len(train_loader)
        train_accuracy /= len(train_loader)
        
        val_loss = 0  # Initialize validation loss
        val_accuracy = 0  # Initialize validation accuracy
        
        # Validation loop
        for x, y in tqdm(val_loader, desc="Validating"):
            y_hat = model(x)  # Forward pass on validation data
            val_loss += loss_fn(y_hat, y).value  # Accumulate validation loss
            val_accuracy += (y_hat.argmax(axis=1) == y.argmax(axis=1)).mean()  # Compute batch accuracy
        
        # Compute average validation loss and accuracy
        val_loss /= len(val_loader)
        val_accuracy /= len(val_loader)
        
        # Track the best validation loss and accuracy
        best_val_loss = min(best_val_loss, val_loss)
        best_val_accuracy = max(best_val_accuracy, val_accuracy)
        
        # Log validation metrics
        wandb.log(
            {
                "val/loss": val_loss,
                "val/acc": val_accuracy,
                "val/best_loss": best_val_loss,
                "val/best_acc": best_val_accuracy,
            }
        )
        
        # Print epoch summary
        print(
            f"Epoch {epoch + 1}/{n_epochs} |"
            f" Train Loss: {train_loss:.4f} | Train Accuracy: {train_accuracy:.4f} |"
            f" Val Loss: {val_loss:.4f} | Val Accuracy: {val_accuracy:.4f}"
        )
    
    return model


def main(args):
    """
    Main function to train, evaluate, and log a neural network model.
    
    Args:
        args: A namespace containing hyperparameters and configurations.
    """
    # Initialize hyperparameters from command-line arguments
    hyperparameters = HyperParameters(
        hidden_activations=args.activation,
        init_strategy=args.weight_init,
        loss_fn=args.loss,
        optimizer=args.optimizer,
        learning_rate=args.learning_rate,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        beta_1=args.beta1,
        beta_2=args.beta2,
        hidden_sizes=args.hidden_sizes,
        epsilon=args.epsilon
    )
    
    # Create a unique run name for logging
    run_name = f"hl_{len(hyperparameters.hidden_sizes)}_bs_{hyperparameters.batch_size}_ac_{hyperparameters.hidden_activations}_opt_{hyperparameters.optimizer}_loss_{hyperparameters.loss_fn}"
    wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=run_name)
    
    # Load dataset
    dataset = Dataset(args.dataset)
    
    # Initialize data loaders for training, validation, and testing
    train_loader = DataLoader(dataset.x_train, dataset.y_train, batch_size=hyperparameters.batch_size)
    val_loader = DataLoader(dataset.x_dev, dataset.y_dev, batch_size=hyperparameters.batch_size)
    test_loader = DataLoader(dataset.x_test, dataset.y_test, batch_size=hyperparameters.batch_size)
    
    # Initialize neural network model
    model = NeuralNetwork(
        train_loader.x.shape[1], train_loader.y.shape[1], hyperparameters
    )
    
    # Define optimizer based on the chosen strategy
    optimizer = {
        "sgdm": SGDM(
            model.parameters,
            hyperparameters.learning_rate,
            hyperparameters.beta_1,
            hyperparameters.weight_decay,
            hyperparameters.epsilon,
        ),
        "nag": SGDM(
            model.parameters,
            hyperparameters.learning_rate,
            hyperparameters.beta_1,
            hyperparameters.weight_decay,
            hyperparameters.epsilon,
            nesterov=True,
        ),
        "adam": AdamW(
            model.parameters,
            hyperparameters.learning_rate,
            hyperparameters.beta_1,
            hyperparameters.beta_2,
            hyperparameters.weight_decay,
            epsilon=hyperparameters.epsilon,
        ),
        "nadam": AdamW(
            model.parameters,
            hyperparameters.learning_rate,
            hyperparameters.beta_1,
            hyperparameters.beta_2,
            hyperparameters.weight_decay,
            nesterov=True,
            epsilon=hyperparameters.epsilon,
        ),
        "rmsprop": RMSProp(
            model.parameters,
            hyperparameters.learning_rate,
            hyperparameters.beta_1,
            hyperparameters.weight_decay,
            hyperparameters.epsilon,
        ),
    }.get(
        hyperparameters.optimizer,
        AdamW(
            model.parameters,
            hyperparameters.learning_rate,
            hyperparameters.beta_1,
            hyperparameters.beta_2,
            hyperparameters.epsilon,
            hyperparameters.weight_decay,
        ),
    )
    
    # Define loss function based on the chosen strategy
    loss_fn = {
        "mse": MSELoss,
        "nll": NLLLoss,
    }.get(hyperparameters.loss_fn, NLLLoss)
    loss_fn = loss_fn(model)
    
    # Train the model
    model = train(
        model, train_loader, val_loader, optimizer, loss_fn, hyperparameters.n_epochs
    )
    
    # Evaluate the model on the test set
    test_loss = 0
    test_accuracy = 0
    true_labels, pred_labels = [], []
    for x, y in tqdm(test_loader, desc="Testing"):
        y_hat = model(x)
        test_loss += loss_fn(y_hat, y).value
        test_accuracy += (y_hat.argmax(axis=1) == y.argmax(axis=1)).mean()
        true_labels.extend(y.argmax(axis=1).tolist())
        pred_labels.extend(y_hat.argmax(axis=1).tolist())
    
    # Compute average test loss and accuracy
    test_loss /= len(test_loader)
    test_accuracy /= len(test_loader)
    
    # Log test metrics to WandB
    wandb.log({"test/loss": test_loss, "test/acc": test_accuracy})
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
    
    # Convert labels to numpy arrays for confusion matrix computation
    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)
    
    # Compute and normalize the confusion matrix
    cm = confusion_matrix(true_labels, pred_labels, normalize="true")
    
    # Define class labels for the dataset
    labels = [
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", 
        "Shirt", "Sneaker", "Bag", "Ankle boot"
    ]
    
    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    
    # Set labels and title for the plot
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix")
    plt.xticks(rotation=45, ha="right")
    
    # Log the confusion matrix plot to WandB
    wandb.log({"test/confusion_matrix": fig})
    
    # Close the plot to free memory
    plt.close(fig)
    
    # Finish the WandB run
    wandb.finish()


def sweep(sweep_config):
    """
    Runs a hyperparameter sweep using configurations from a JSON file.
    
    This function loads a sweep configuration from "sweep_config.json", initializes a sweep
    using Weights & Biases (wandb), and iterates over different hyperparameter settings to train 
    and evaluate a neural network model.
    """
    with open(sweep_config) as f:
        sweep_config = json.load(f)

    def train_sweep(config=None):
        """
        Trains a neural network model using hyperparameters provided by the sweep configuration.
        
        This function is executed as part of the sweep and performs training, validation, and 
        testing for each hyperparameter setting. The results are logged to wandb.
        
        Args:
            config (dict, optional): A dictionary containing the hyperparameter settings for the current run.
        """
        with wandb.init(config=config):
            config = wandb.config
            hyperparameters = HyperParameters(
                hidden_activations=config.hidden_activations,
                init_strategy=config.init_strategy,
                loss_fn=config.loss_fn,
                optimizer=config.optimizer,
                learning_rate=config.learning_rate,
                n_epochs=config.n_epochs,
                batch_size=config.batch_size,
                weight_decay=config.weight_decay,
                beta_1=config.beta_1,
                beta_2=config.beta_2,
                hidden_sizes=config.hidden_sizes,
                epsilon=config.epsilon,
            )
            
            run_name = f"hl_{len(hyperparameters.hidden_sizes)}_bs_{hyperparameters.batch_size}_ac_{hyperparameters.hidden_activations}_opt_{hyperparameters.optimizer}_loss_{hyperparameters.loss_fn}"
            wandb.run.name = run_name
            
            # Load dataset
            dataset = Dataset("fashion_mnist")
            
            # Visualize some training images
            utils.plot_images(
                dataset.x_train,
                dataset.y_train,
                [
                    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", 
                    "Shirt", "Sneaker", "Bag", "Ankle boot"
                ],
                use_wandb=True,
            )
            
            # Initialize data loaders
            train_loader = DataLoader(dataset.x_train, dataset.y_train, batch_size=hyperparameters.batch_size)
            val_loader = DataLoader(dataset.x_dev, dataset.y_dev, batch_size=hyperparameters.batch_size)
            test_loader = DataLoader(dataset.x_test, dataset.y_test, batch_size=hyperparameters.batch_size)
            
            # Initialize the neural network model
            model = NeuralNetwork(train_loader.x.shape[1], train_loader.y.shape[1], hyperparameters)
            
            # Select optimizer based on user configuration
            optimizer = {
                "sgd": SGDM(model.parameters, hyperparameters.learning_rate, 0.0, hyperparameters.weight_decay, hyperparameters.epsilon),
                "sgdm": SGDM(model.parameters, hyperparameters.learning_rate, hyperparameters.beta_1, hyperparameters.weight_decay, hyperparameters.epsilon),
                "nag": SGDM(model.parameters, hyperparameters.learning_rate, hyperparameters.beta_1, hyperparameters.weight_decay, hyperparameters.epsilon, nesterov=True),
                "adam": AdamW(model.parameters, hyperparameters.learning_rate, hyperparameters.beta_1, hyperparameters.beta_2, hyperparameters.weight_decay, epsilon=hyperparameters.epsilon),
                "nadam": AdamW(model.parameters, hyperparameters.learning_rate, hyperparameters.beta_1, hyperparameters.beta_2, hyperparameters.weight_decay, nesterov=True, epsilon=hyperparameters.epsilon),
                "rmsprop": RMSProp(model.parameters, hyperparameters.learning_rate, hyperparameters.beta_1, hyperparameters.weight_decay, hyperparameters.epsilon),
            }.get(hyperparameters.optimizer, AdamW(model.parameters, hyperparameters.learning_rate, hyperparameters.beta_1, hyperparameters.beta_2, hyperparameters.epsilon, hyperparameters.weight_decay))
            
            # Select loss function
            loss_fn = {"mse": MSELoss, "nll": NLLLoss}.get(hyperparameters.loss_fn, NLLLoss)
            loss_fn = loss_fn(model)
            
            # Ensure softmax activation for MSE loss
            if hyperparameters.loss_fn == "mse":
                model.output_activation = Softmax()
            
            # Train the model
            model = train(model, train_loader, val_loader, optimizer, loss_fn, hyperparameters.n_epochs)
            
            # Evaluate the model on the test set
            test_loss = 0
            test_accuracy = 0
            for x, y in tqdm(test_loader, desc="Testing"):
                y_hat = model(x)
                test_loss += loss_fn(y_hat, y).value
                test_accuracy += (y_hat.argmax(axis=1) == y.argmax(axis=1)).mean()
            test_loss /= len(test_loader)
            test_accuracy /= len(test_loader)
            
            # Log test results
            wandb.log({"test/loss": test_loss, "test/acc": test_accuracy})
            print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
    
    # Initialize and run the hyperparameter sweep
    sweep_id = wandb.sweep(sweep_config, project="da6401_assignment1")
    wandb.agent(sweep_id, train_sweep, count=75)


def parse_args():
    parser = ArgumentParser()

    parser.add_argument("-wp", "--wandb_project", type=str, default="da6401_assignment1", help="Wandb project to use for logging")
    parser.add_argument("-we", "--wandb_entity", type=str, default="iamunr4v31", help="Wandb entity to use for logging")
    parser.add_argument("-d", "--dataset", type=str, default="fashion_mnist", help="Dataset to use for training")
    parser.add_argument("-e", "--epochs", type=int, default=20, help="Number of epochs to train for")
    parser.add_argument("-b", "--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("-l", "--loss", type=str, default="nll", help="Loss function to use")
    parser.add_argument("-o", "--optimizer", type=str, default="nadam", help="Optimizer to use")
    parser.add_argument("-lr", "--learning_rate", type=float, default=1.759418900444131e-4, help="Learning Rate for Optimizers")
    parser.add_argument("-beta1", "--beta1", type=float, default=0.95, help="Use as momentum for SGDM, RMSProp, and Nesterov")
    parser.add_argument("-beta2", "--beta2", type=float, default=0.999, help="Beta 2 for Adam and Nadam")
    parser.add_argument("-eps", "--epsilon", type=float, default=1.0024073065576981e-7, help="Numerical Stability Constant")
    parser.add_argument("-w_d", "--weight_decay", type=float, default=5.322763180247343e-4, help="Use as weight decay (regularization coefficient)")
    parser.add_argument("-w_i", "--weight_init", type=str, default="xavier", help="Weight initialization strategy", choices=["he", "xavier", "normal"])
    parser.add_argument("-sz", "--hidden_sizes", type=int, nargs="+", default=[2048, 1024], help="Hidden layer sizes, Pass a list separated by spaces")
    parser.add_argument("-ac", "--activation", type=str, default="tanh", help="Activation function for hidden layers")
    parser.add_argument("--do_sweep", action="store_true", help="Flag to indicate whether to run a hyperparameter sweep")
    parser.add_argument("--sweep_config", type=str, default="sweep_config.json", help="Path to the sweep configuration file")    

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.do_sweep:
        sweep(args.sweep_config)
    else:
        main(args)