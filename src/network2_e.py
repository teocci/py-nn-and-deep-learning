"""
Created by Teocci.
Author: teocci@yandex.com
Date: 2025-2월-07
"""

"""network2_e.py
~~~~~~~~~~~~~~
An improved version of network.py, implementing the stochastic
gradient descent learning algorithm for a feedforward neural network.
Improvements include the addition of the cross-entropy cost function,
regularization, and better initialization of network weights. Note
that I have focused on making the code simple, easily readable, and
easily modifiable. It is not optimized, and omits many desirable
features.
"""
# Standard library
import json
import logging
import random
import sys
from typing import List, Tuple

# Third-party libraries
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


#### Define the quadratic and cross-entropy cost functions
class QuadraticCost:
    @staticmethod
    def fn(a: np.ndarray, y: np.ndarray) -> float:
        """Return the cost associated with an output `a` and desired output `y`."""
        return 0.5 * np.linalg.norm(a - y) ** 2

    @staticmethod
    def delta(z: np.ndarray, a: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Return the error delta from the output layer."""
        return (a - y) * sigmoid_prime(z)


class CrossEntropyCost:
    @staticmethod
    def fn(a: np.ndarray, y: np.ndarray) -> float:
        """Return the cost associated with an output `a` and desired output `y`.
        Note that np.nan_to_num is used to ensure numerical stability.

        """
        return np.sum(np.nan_to_num(-y * np.log(a) - (1 - y) * np.log(1 - a)))

    @staticmethod
    def delta(z: np.ndarray, a: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Return the error delta from the output layer."""
        return a - y


#### Main Network class
class Network:
    def __init__(self, sizes: List[int], cost=CrossEntropyCost):
        """The list `sizes` contains the number of neurons in the respective
        layers of the network. For example, if the list was [2, 3, 1]
        then it would be a three-layer network, with the first layer
        containing 2 neurons, the second layer 3 neurons, and the
        third layer 1 neuron. The biases and weights for the network
        are initialized randomly, using `self.default_weight_initializer`.

        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.default_weight_initializer()
        self.cost = cost

    def default_weight_initializer(self):
        """Initialize each weight using a Gaussian distribution with mean 0
        and standard deviation 1 over the square root of the number of
        weights connecting to the same neuron. Initialize the biases
        using a Gaussian distribution with mean 0 and standard deviation 1.

        """
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [
            np.random.randn(y, x) / np.sqrt(x)
            for x, y in zip(self.sizes[:-1], self.sizes[1:])
        ]

    def large_weight_initializer(self):
        """Initialize the weights using a Gaussian distribution with mean 0
        and standard deviation 1. Initialize the biases using a
        Gaussian distribution with mean 0 and standard deviation 1.

        """
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [
            np.random.randn(y, x)
            for x, y in zip(self.sizes[:-1], self.sizes[1:])
        ]

    def feedforward(self, a: np.ndarray) -> np.ndarray:
        """Return the output of the network if `a` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def sgd(
            self,
            training_data: List[Tuple[np.ndarray, np.ndarray]],
            epochs: int,
            mini_batch_size: int,
            eta: float,
            lmbda: float = 0.0,
            evaluation_data: List[Tuple[np.ndarray, np.ndarray]] = None,
            monitor_evaluation_cost: bool = False,
            monitor_evaluation_accuracy: bool = False,
            monitor_training_cost: bool = False,
            monitor_training_accuracy: bool = False,
            is_sub_training: bool = False,
    ) -> Tuple[List[float], List[int], List[float], List[int]]:
        """Train the neural network using mini-batch stochastic gradient descent."""
        if evaluation_data:
            n_data = len(evaluation_data)
        n = len(training_data)
        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []
        label = "Step" if is_sub_training else "Epoch"

        for ei in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k: k + mini_batch_size]
                for k in range(0, n, mini_batch_size)
            ]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta, lmbda, len(training_data))

            logging.info(f"{label} {ei + 1} training complete")

            if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda)
                training_cost.append(cost)
                logging.info(f"Cost on training data: {cost}")

            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data, convert=True)
                training_accuracy.append(accuracy)
                logging.info(f"Accuracy on training data: {accuracy} / {n}")

            if monitor_evaluation_cost and evaluation_data:
                cost = self.total_cost(evaluation_data, lmbda, convert=True)
                evaluation_cost.append(cost)
                logging.info(f"Cost on evaluation data: {cost}")

            if monitor_evaluation_accuracy and evaluation_data:
                accuracy = self.accuracy(evaluation_data)
                evaluation_accuracy.append(accuracy)
                logging.info(f"Accuracy on evaluation data: {accuracy} / {n_data}")

        return evaluation_cost, evaluation_accuracy, training_cost, training_accuracy

    def update_mini_batch(
            self,
            mini_batch: List[Tuple[np.ndarray, np.ndarray]],
            eta: float,
            lmbda: float,
            n: int,
    ):
        """Update the network's weights and biases by applying gradient
        descent using backpropagation to a single mini batch.

        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.weights = [
            (1 - eta * (lmbda / n)) * w - (eta / len(mini_batch)) * nw
            for w, nw in zip(self.weights, nabla_w)
        ]
        self.biases = [
            b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)
        ]

    def backprop(self, x: np.ndarray, y: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Return a tuple `(nabla_b, nabla_w)` representing the gradient for the cost function."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # Feedforward
        activation = x
        activations = [x]  # Store all activations, layer by layer
        zs = []  # Store all z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # Backward pass
        delta = self.cost.delta(zs[-1], activations[-1], y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].T)

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].T, delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].T)

        return nabla_b, nabla_w

    def accuracy(self, data: List[Tuple[np.ndarray, np.ndarray]], convert: bool = False) -> int:
        """Return the number of inputs in `data` for which the neural network outputs the correct result."""

        def get_label(y):
            """Convert `y` to an integer label if `convert=True`."""
            return np.argmax(y) if convert else y

        results = [(np.argmax(self.feedforward(x)), get_label(y)) for x, y in data]
        return sum(int(x == y) for x, y in results)

    def total_cost(self, data: List[Tuple[np.ndarray, np.ndarray]], lmbda: float, convert: bool = False) -> float:
        """Return the total cost for the data set `data`."""
        cost = 0.0
        for x, y in data:
            a = self.feedforward(x)
            if convert:
                y = vectorized_result(y)
            cost += self.cost.fn(a, y) / len(data)

        cost += 0.5 * (lmbda / len(data)) * sum(np.linalg.norm(w) ** 2 for w in self.weights)
        return cost

    def save(self, filename: str):
        """Save the neural network to the file `filename`."""
        data = {
            "sizes": self.sizes,
            "weights": [w.tolist() for w in self.weights],
            "biases": [b.tolist() for b in self.biases],
            "cost": str(self.cost.__name__),
        }
        with open(filename, "w") as f:
            json.dump(data, f)


def load(filename: str) -> Network:
    """Load a neural network from the file `filename`. Returns an instance of Network."""
    with open(filename, "r") as f:
        data = json.load(f)
    cost = getattr(sys.modules[__name__], data["cost"])
    net = Network(data["sizes"], cost=cost)
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net


#### Miscellaneous functions
def vectorized_result(j: int) -> np.ndarray:
    """Return a 10-dimensional unit vector with a 1.0 in the j'th position."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


def sigmoid(z: np.ndarray) -> np.ndarray:
    """The sigmoid function."""
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z: np.ndarray) -> np.ndarray:
    """Derivative of the sigmoid function."""
    return sigmoid(z) * (1 - sigmoid(z))
