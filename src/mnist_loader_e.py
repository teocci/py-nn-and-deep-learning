"""
Created by Teocci.
Author: teocci@yandex.com
Date: 2025-2월-07
"""

"""
mnist_loader
~~~~~~~~~~~~
A library to load the MNIST image data. For details of the data structures 
that are returned, see the docstrings for ``load_data`` and ``load_data_wrapper``. 
In practice, ``load_data_wrapper`` is the function usually called by our neural 
network code.
"""

import gzip
import pickle
from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray

# Type aliases for clarity
ImageData = NDArray[np.float64]
LabelData = NDArray[np.int64]
ProcessedData = Tuple[NDArray[np.float64], NDArray[np.float64]]

# Constants
IMAGE_SIZE = 784  # 28 * 28 pixels
LABELS_SIZE = 10  # Digits 0-9


def load_data() -> Tuple[
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray]
]:
    """Return the MNIST data as a tuple containing the training data,
    the validation data, and the test data.

    The ``training_data`` is returned as a tuple with two entries:
    - The first entry contains the actual training images (50,000 entries).
      Each entry is a numpy ndarray with 784 values representing the pixels.
    - The second entry contains the digit labels (0-9) for the corresponding images.

    The ``validation_data`` and ``test_data`` are similar but contain only 10,000 images each.
    """
    with gzip.open('../data/mnist.pkl.gz', 'rb') as f:
        return pickle.load(f, encoding='latin1')


def preprocess_images(images: ImageData) -> NDArray[np.float64]:
    """Reshape each image into a column vector."""
    return images.reshape(-1, IMAGE_SIZE, 1)


def preprocess_labels(labels: LabelData) -> NDArray[np.float64]:
    """Convert labels to either one-hot vectors or integers."""
    return np.eye(LABELS_SIZE)[labels].reshape(-1, LABELS_SIZE, 1)


def load_data_wrapper() -> Tuple[
    List[Tuple[np.ndarray, np.ndarray]],
    List[Tuple[np.ndarray, int]],
    List[Tuple[np.ndarray, int]]
]:
    """Return a tuple containing ``(training_data, validation_data, test_data)``.

    This function modifies the format of the data returned by ``load_data`` to be more
    convenient for use in neural network implementations.

    - ``training_data``: A list of 50,000 tuples ``(x, y)``, where:
      - ``x`` is a 784-dimensional numpy array representing the input image.
      - ``y`` is a 10-dimensional unit vector representing the correct digit.

    - ``validation_data`` and ``test_data``: Lists of 10,000 tuples ``(x, y)``, where:
      - ``x`` is a 784-dimensional numpy array representing the input image.
      - ``y`` is the integer label (0-9) for the image.
    """
    tr_d, va_d, te_d = load_data()

    # Preprocess training data
    train_images = preprocess_images(tr_d[0])
    train_labels = preprocess_labels(tr_d[1])
    training_data = list(zip(train_images, train_labels))

    # Preprocess validation and test data
    validation_images = preprocess_images(va_d[0])
    validation_data = list(zip(validation_images, va_d[1]))

    test_images = preprocess_images(te_d[0])
    test_data = list(zip(test_images, te_d[1]))

    return training_data, validation_data, test_data
