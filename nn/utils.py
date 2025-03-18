import numpy as np
from keras.datasets import mnist, cifar10, fashion_mnist
from sklearn.model_selection import train_test_split
from typing import Tuple

class Dataset:
    """
    A class for loading and managing standard image datasets.
    
    Supports MNIST, CIFAR-10, and Fashion-MNIST datasets. 
    Automatically splits the training data into training and development sets.
    
    Args:
        name (str): Name of the dataset to load ('mnist', 'cifar10', or 'fashion_mnist').
        seed (int, optional): Random seed for reproducible data splitting. Defaults to 42.
    
    Raises:
        ValueError: If the requested dataset name is not supported.
    """
    def __init__(self, name: str, seed: int = 42) -> None:
        self.name = name
        self.load_data(seed)

    def load_data(self, seed) -> None:
        """
        Load the specified dataset and split it into train, dev, and test sets.
        
        Args:
            seed (int): Random seed for reproducible data splitting.
            
        Raises:
            ValueError: If the dataset name is not supported.
        """
        # Load the appropriate dataset based on the name
        if self.name == 'mnist':
            (x_train, y_train), (x_test, y_test) = mnist.load_data()
        elif self.name == 'cifar10':
            (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        elif self.name == 'fashion_mnist':
            (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        else:
            raise ValueError(f"Dataset {self.name} is not supported.")
        
        # Split training data to create a development set (10% of training data)
        x_train, x_dev, y_train, y_dev = train_test_split(
            x_train, y_train, test_size=0.1, random_state=seed
        )
        
        # Store all data splits as instance variables
        self.x_train, self.y_train = x_train, y_train
        self.x_dev, self.y_dev = x_dev, y_dev
        self.x_test, self.y_test = x_test, y_test

    def get_splits(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get all data splits.
        
        Returns:
            tuple: A tuple containing:
                - (x_train, y_train): Training data and labels
                - (x_dev, y_dev): Development data and labels
                - (x_test, y_test): Test data and labels
        """
        return (self.x_train, self.y_train), (self.x_dev, self.y_dev), (self.x_test, self.y_test)
 
class DataLoader:
    """
    Batch data loader for training neural networks.
    
    Provides an iterator interface for loading batches of data during training.
    Handles data normalization, flattening, and one-hot encoding of labels.
    
    Args:
        x (np.ndarray): Input features.
        y (np.ndarray): Target labels (integers).
        batch_size (int, optional): Number of samples per batch. Defaults to 32.
        shuffle (bool, optional): Whether to shuffle data at the start of each iteration.
                                 Defaults to True.
        flatten (bool, optional): Whether to flatten the input features. Defaults to True.
    """
    def __init__(self, x, y, batch_size=32, shuffle=True, flatten=True) -> None:
        # Flatten the input data if requested (e.g., for MLPs)
        self.x = x if not flatten else x.reshape(x.shape[0], -1)
        # Normalize pixel values to [0, 1] range
        self.x = self.x / 255.0
        # Convert integer labels to one-hot encoded vectors
        self.y = np.eye(np.max(y) + 1)[y]
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        """
        Initialize the iterator.
        
        Prepares for iteration by setting up indices and optionally shuffling them.
        
        Returns:
            DataLoader: Self reference to enable iteration.
        """
        self.n = self.x.shape[0]  # Total number of samples
        self.indices = np.arange(self.n)  # Create array of indices
        if self.shuffle:
            np.random.shuffle(self.indices)  # Shuffle indices if requested
        self.i = 0  # Initialize batch pointer
        return self

    def __next__(self):
        """
        Get the next batch of data.
        
        Returns:
            tuple: (batch_x, batch_y) containing the next batch of features and labels.
            
        Raises:
            StopIteration: When all batches have been processed.
        """
        # Check if we've reached the end of the dataset
        if self.i >= self.n:
            raise StopIteration
        
        # Get indices for the current batch
        batch_indices = self.indices[self.i:self.i+self.batch_size]
        # Extract the batch data using these indices
        batch_x, batch_y = self.x[batch_indices], self.y[batch_indices]
        # Move pointer to the next batch
        self.i += self.batch_size
        
        return batch_x, batch_y
    
    def __len__(self):
        """
        Get the total number of batches.
        
        Returns:
            int: Number of batches in the dataset.
        """
        return int(np.ceil(self.n / self.batch_size))
