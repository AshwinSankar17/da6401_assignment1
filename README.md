# DA6401 Assignment 1: Neural Networks from scratch

## Instructions

The code is written in Python 3.10.9. The following libraries are used:
- Numpy
- Matplotlib
- Scikit-learn
- Wandb
- Keras

It is advised to run the code in a virtual environment. The requirements.txt file contains the list of libraries used. Use the following command to install the libraries.

```bash
pip install -r requirements.txt

```

The repository implements a module which closely imitates the API of PyTorch. The module is used to implement the neural network. Use train.py to train the model. The hyperparameters can be changed though command line arguments. The accuracy is printed on the console.

|           Name           | Description                                                                   |
| :----------------------: | :-----------------------------------------------------------------------------|
| `-wp`, `--wandb_project` | Project name used to track experiments in Weights & Biases dashboard          |
|  `-we`, `--wandb_entity` | Wandb Entity used to track experiments in the Weights & Biases dashboard.     |
|     `-d`, `--dataset`    | choices:  ["mnist", "fashion_mnist", "cifar10"]                               |
|     `-e`, `--epochs`     | Number of epochs to train neural network.                                     |
|   `-b`, `--batch_size`   | Batch size used to train neural network.                                      |
|      `-l`, `--loss`      | choices:  ["mse", "cross_entropy"]                                            |
|    `-o`, `--optimizer`   | choices:  ["sgd", "sgdm", "nag", "rmsprop", "adam", "nadam"]                  |
| `-lr`, `--learning_rate` | Learning rate used to optimize model parameters                               |
|    `-beta1`, `--beta1`   | Beta1 used by adam and nadam optimizers. Used as momentum for other optimizers|
|    `-beta2`, `--beta2`   | Beta2 used by adam and nadam optimizers.                                      |
|    `-eps`, `--epsilon`   | Epsilon used by optimizers.                                                   |
| `-w_d`, `--weight_decay`          | Weight decay used by optimizers.                                     |
|  `-w_i`, `--weight_init` | choices:  ["xavier", "he", "normal"]                                          |
|  `-nhl`, `--num_layers`  | Number of hidden layers used in feedforward neural network.                   |
|  `-sz`, `--hidden_size`  | Number of hidden neurons in a feedforward layer.                              |
|   `-ac`, `--activation`  | choices:  ["sigmoid", "tanh", "ReLU"]                                         |

## Example Run Instance

```bash
python train.py --wandb_project "best_val_acc_project" --wandb_entity "your_entity"
```

## Code Organization

```bash
da6401_assignment1/
├── sweep_config.json
├── sweep_config.yaml
├── train.py
└── utils.py
├── nn/
│   ├── utils.py
│   ├── __init__.py
│   ├── activations.py
│   ├── losses.py
│   ├── core.py
│   ├── optim.py
│   └── layers.py
```