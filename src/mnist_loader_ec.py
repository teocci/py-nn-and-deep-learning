"""
Created by Teocci.
Author: teocci@yandex.com
Date: 2025-2월-07
"""

"""
mnist_loader
~~~~~~~~~~~~
A highly optimized MNIST data loader using NumPy vectorization.
"""

import gzip
import pickle
from pathlib import Path
from typing import Tuple

import numpy as np
from numpy.typing import NDArray

# Type aliases for clarity
ImageData = NDArray[np.float64]
LabelData = NDArray[np.int64]
ProcessedData = Tuple[NDArray[np.float64], NDArray[np.float64]]


class MNISTLoader:
    """A class to handle MNIST data loading with optimized NumPy operations."""

    def __init__(self, data_path: str = '../data/mnist.pkl.gz'):
        self.data_path = Path(data_path)

    @staticmethod
    def _validate_data(data: Tuple[Tuple[ImageData, LabelData], ...]) -> None:
        """Validate the structure and dimensions of loaded MNIST data."""
        expected_shapes = {
            'training': (50000, 784),
            'validation': (10000, 784),
            'test': (10000, 784)
        }

        for (images, labels), (name, shape) in zip(data, expected_shapes.items()):
            if images.shape[0] != shape[0] or images.shape[1] != shape[1]:
                raise ValueError(
                    f"Invalid {name} data shape. Expected {shape}, got {images.shape}"
                )

    def load_raw_data(self) -> Tuple[Tuple[ImageData, LabelData], ...]:
        """Load raw MNIST data from the gzipped pickle file."""
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"MNIST data file not found at {self.data_path}. "
                "Please download the data first."
            )

        with gzip.open(self.data_path, 'rb') as f:
            try:
                data = pickle.load(f, encoding='latin1')
                self._validate_data(data)
                return data
            except (pickle.UnpicklingError, EOFError) as e:
                raise ValueError(f"Error loading MNIST data: {e}")

    @staticmethod
    def _process_images(images: ImageData) -> NDArray[np.float64]:
        """Process images using efficient NumPy reshape."""
        return images.reshape(-1, 784, 1)

    @staticmethod
    def _vectorize_labels(labels: LabelData) -> NDArray[np.float64]:
        """Vectorize labels using efficient NumPy operations."""
        return np.eye(10)[labels].reshape(-1, 10, 1)

    def load_processed_data(self) -> Tuple[ProcessedData, ProcessedData, ProcessedData]:
        """Load and process MNIST data optimized for neural network training."""
        training_data, validation_data, test_data = self.load_raw_data()

        # Process training data
        train_images = self._process_images(training_data[0])
        train_labels = self._vectorize_labels(training_data[1])

        # Process validation and test data
        validation_images = self._process_images(validation_data[0])
        test_images = self._process_images(test_data[0])

        return ((train_images, train_labels),
                (validation_images, validation_data[1]),
                (test_images, test_data[1]))


# Convenience functions
def load_data():
    """Load raw MNIST data."""
    return MNISTLoader().load_raw_data()


def load_data_wrapper():
    """Load processed MNIST data."""
    return MNISTLoader().load_processed_data()
