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
    def __init__(
        self, input_size: int, output_size: int, hyperparameters: HyperParameters
    ) -> None:
        super().__init__()
        self.output_activation = LogSoftmax()
        self.hyperparameters = hyperparameters
        self.layers = []

        fc_in = Linear(
            input_size, hyperparameters.hidden_sizes[0], hyperparameters.init_strategy
        )
        self.activation = {
            "relu": ReLU(),
            "sigmoid": Sigmoid(),
            "tanh": Tanh(),
        }.get(hyperparameters.hidden_activations, ReLU())
        self.layers.append(fc_in)

        for i in range(1, len(hyperparameters.hidden_sizes)):
            fc = Linear(
                hyperparameters.hidden_sizes[i - 1],
                hyperparameters.hidden_sizes[i],
                hyperparameters.init_strategy,
            )
            self.layers.append(fc)

        fc_out = Linear(
            hyperparameters.hidden_sizes[-1], output_size, hyperparameters.init_strategy
        )
        self.layers.append(fc_out)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.activation(x)
        x = self.layers[-1](x)
        return self.output_activation(x)


def train(model, train_loader, val_loader, optimizer, loss_fn, n_epochs):
    best_val_loss = float("inf")
    best_val_accuracy = 0
    for epoch in range(n_epochs):
        train_loss = 0.0
        train_accuracy = 0.0
        for step_idx, (x, y) in tqdm(
            enumerate(train_loader), desc="Training", total=len(train_loader)
        ):
            optimizer.zero_grad()
            y_hat = model(x)
            loss = loss_fn(y_hat, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.value
            train_accuracy += (y_hat.argmax(axis=1) == y.argmax(axis=1)).mean()
            wandb.log(
                {
                    "train/loss": loss.value,
                    "train/acc": (y_hat.argmax(axis=1) == y.argmax(axis=1)).mean(),
                }
            )
        train_loss /= len(train_loader)
        train_accuracy /= len(train_loader)
        val_loss = 0
        val_accuracy = 0
        for x, y in tqdm(val_loader, "Validating"):
            y_hat = model(x)
            val_loss += loss_fn(y_hat, y).value
            val_accuracy += (y_hat.argmax(axis=1) == y.argmax(axis=1)).mean()
        val_loss /= len(val_loader)
        val_accuracy /= len(val_loader)
        best_val_loss = min(best_val_loss, val_loss)
        best_val_accuracy = max(best_val_accuracy, val_accuracy)
        wandb.log(
            {
                "val/loss": val_loss,
                "val/acc": val_accuracy,
                "val/best_loss": best_val_loss,
                "val/best_acc": best_val_accuracy,
            }
        )
        print(
            f"Epoch {epoch + 1}/{n_epochs} |"
            f" Train Loss: {train_loss:.4f} | Train Accuracy: {train_accuracy:.4f} |"
            f" Val Loss: {val_loss:.4f} | Val Accuracy: {val_accuracy:.4f}"
        )
    return model


def main(args):
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
    run_name = f"hl_{len(hyperparameters.hidden_sizes)}_bs_{hyperparameters.batch_size}_ac_{hyperparameters.hidden_activations}_opt_{hyperparameters.optimizer}_loss_{hyperparameters.loss_fn}"
    wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=run_name)
    dataset = Dataset(args.dataset)
    # utils.plot_images(dataset.x_train, dataset.y_train)
    train_loader = DataLoader(dataset.x_train, dataset.y_train, batch_size=hyperparameters.batch_size)
    val_loader = DataLoader(dataset.x_dev, dataset.y_dev, batch_size=hyperparameters.batch_size)
    test_loader = DataLoader(dataset.x_test, dataset.y_test, batch_size=hyperparameters.batch_size)
    model = NeuralNetwork(
        train_loader.x.shape[1], train_loader.y.shape[1], hyperparameters
    )
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
    loss_fn = {
        "mse": MSELoss,
        "nll": NLLLoss,
    }.get(hyperparameters.loss_fn, NLLLoss)
    loss_fn = loss_fn(model)
    model = train(
        model, train_loader, val_loader, optimizer, loss_fn, hyperparameters.n_epochs
    )
    if hyperparameters.loss_fn == "mse":
        model.output_activation = Softmax()
    test_loss = 0
    test_accuracy = 0
    # true_labels, pred_labels = [], []
    for x, y in tqdm(test_loader, desc="Testing"):
        y_hat = model(x)
        test_loss += loss_fn(y_hat, y).value
        test_accuracy += (y_hat.argmax(axis=1) == y.argmax(axis=1)).mean()
        # true_labels.extend(y.argmax(axis=1).tolist())
        # pred_labels.extend(y_hat.argmax(axis=1).tolist())
    test_loss /= len(test_loader)
    test_accuracy /= len(test_loader)
    wandb.log({"test/loss": test_loss, "test/acc": test_accuracy})
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    # true_labels = np.array(true_labels)
    # pred_labels = np.array(pred_labels)

    # # Compute confusion matrix
    # cm = confusion_matrix(true_labels, pred_labels, normalize="true") * 100

    # # Define class labels
    # labels = [
    #     "T-shirt/top",
    #     "Trouser",
    #     "Pullover",
    #     "Dress",
    #     "Coat",
    #     "Sandal",
    #     "Shirt",
    #     "Sneaker",
    #     "Bag",
    #     "Ankle boot",
    # ]

    # fig, ax = plt.subplots(figsize=(8, 6))
    # sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)

    # # Set labels and title
    # ax.set_xlabel("Predicted Label")
    # ax.set_ylabel("True Label")
    # ax.set_title("Confusion Matrix")
    
    # # Log the figure to WandB
    # wandb.log({"test/confusion_matrix": fig})
    # # Close the plot to free memory
    # plt.close(fig)


    wandb.finish()


def sweep():
    with open("sweep_config.json") as f:
        sweep_config = json.load(f)

    def train_sweep(config=None):
        global sweep_count
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
            dataset = Dataset("fashion_mnist")

            utils.plot_images(
                dataset.x_train,
                dataset.y_train,
                [
                    "T-shirt/top",
                    "Trouser",
                    "Pullover",
                    "Dress",
                    "Coat",
                    "Sandal",
                    "Shirt",
                    "Sneaker",
                    "Bag",
                    "Ankle boot",
                ],
                use_wandb=True,
            )
            train_loader = DataLoader(
                dataset.x_train, dataset.y_train, batch_size=hyperparameters.batch_size
            )
            val_loader = DataLoader(
                dataset.x_dev, dataset.y_dev, batch_size=hyperparameters.batch_size
            )
            test_loader = DataLoader(
                dataset.x_test, dataset.y_test, batch_size=hyperparameters.batch_size
            )
            model = NeuralNetwork(
                train_loader.x.shape[1], train_loader.y.shape[1], hyperparameters
            )
            optimizer = {
                "sgd": SGDM(
                    model.parameters,
                    hyperparameters.learning_rate,
                    0.0,
                    hyperparameters.weight_decay,
                    hyperparameters.epsilon,
                ),
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
            loss_fn = {
                "mse": MSELoss,
                "nll": NLLLoss,
            }.get(hyperparameters.loss_fn, NLLLoss)
            loss_fn = loss_fn(model)
            if hyperparameters.loss_fn == "mse":
                model.output_activation = Softmax()
            model = train(
                model,
                train_loader,
                val_loader,
                optimizer,
                loss_fn,
                hyperparameters.n_epochs,
            )
            test_loss = 0
            test_accuracy = 0
            for x, y in tqdm(test_loader, desc="Testing"):
                y_hat = model(x)
                test_loss += loss_fn(y_hat, y).value
                test_accuracy += (y_hat.argmax(axis=1) == y.argmax(axis=1)).mean()
            test_loss /= len(test_loader)
            test_accuracy /= len(test_loader)
            wandb.log({"test/loss": test_loss, "test/acc": test_accuracy})
            print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

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


    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)
