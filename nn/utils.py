import numpy as np
from keras.datasets import mnist, cifar10, fashion_mnist
from sklearn.model_selection import train_test_split
from typing import Tuple

class Dataset:
    def __init__(self, name: str, seed: int = 42) -> None:
        self.name = name
        self.load_data(seed)

    def load_data(self, seed) -> None:
        if self.name == 'mnist':
            (x_train, y_train), (x_test, y_test) = mnist.load_data()
        elif self.name == 'cifar10':
            (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        elif self.name == 'fashion_mnist':
            (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        else:
            raise ValueError(f"Dataset {self.name} is not supported.")
        
        x_train, x_dev, y_train, y_dev = train_test_split(x_train, y_train, test_size=0.1, random_state=seed)
        
        self.x_train, self.y_train = x_train, y_train
        self.x_dev, self.y_dev = x_dev, y_dev
        self.x_test, self.y_test = x_test, y_test

    def get_splits(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return (self.x_train, self.y_train), (self.x_dev, self.y_dev), (self.x_test, self.y_test)
 
class DataLoader:
    def __init__(self, x, y, batch_size=32, shuffle=True, flatten=True) -> None:
        self.x = x if not flatten else x.reshape(x.shape[0], -1)
        self.x = self.x / 255.0
        self.y = np.eye(np.max(y) + 1)[y]
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        self.n = self.x.shape[0]
        self.indices = np.arange(self.n)
        if self.shuffle:
            np.random.shuffle(self.indices)
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.n:
            raise StopIteration
        batch_indices = self.indices[self.i:self.i+self.batch_size]
        batch_x, batch_y = self.x[batch_indices], self.y[batch_indices]
        self.i += self.batch_size
        return batch_x, batch_y
    
    def __len__(self):
        return int(np.ceil(self.n / self.batch_size))