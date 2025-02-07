"""
Created by Teocci.
Author: teocci@yandex.com
Date: 2025-2월-07
"""

"""weight_initialization 
~~~~~~~~~~~~~~~~~~~~~~~~
This program shows how weight initialization affects training.  In
particular, we'll plot out how the classification accuracies improve
using either large starting weights, whose standard deviation is 1, or
the default starting weights, whose standard deviation is 1 over the
square root of the number of input neurons.
"""

# Standard library
import json
import random
from typing import List, Tuple

# Third-party libraries
import matplotlib.pyplot as plt
import numpy as np

# My library
import src.mnist_loader as mnist_loader
import src.network2_e as network2

# Constants
WEIGHT_DEFAULT_KEY = "default_weight_initialization"
WEIGHT_LARGE_KEY = "large_weight_initialization"

NUM_EPOCHS = 30
BATCH_SIZE = 10
LAMBDA = 5.0
SEED_VALUE = 12345678


def main(filename: str = "results.json", n: int = 30, eta: float = 0.1):
    """
    Main function to run the network training and generate plots.

    Args:
        filename (str): Filename to store/load results. Defaults to "results.json".
        n (int): Number of hidden neurons. Defaults to 30.
        eta (float): Learning rate. Defaults to 0.1.

    """
    run_network(filename, n, eta)
    make_plot(filename)


def train_network(
        net,
        training_data,
        validation_data,
        eta,
        initializer="default",
        epochs=30,
        mini_batch_size=10,
        lmbda=5.0
) -> Tuple[List[float], List[int], List[float], List[int]]:
    """
    Helper function to train the network and return the results.

    Args:
        net (Network): The neural network to train.
        training_data (list): The training data.
        validation_data (list): The validation data.
        eta (float): The learning rate.
        initializer (str): The weight initialization strategy. Defaults to "default".
        epochs (int): The number of epochs to train for. Defaults to 30.
        mini_batch_size (int): The size of the mini-batches. Defaults to 10.
        lmbda (float): The regularization parameter. Defaults to 5.0.

    Returns:
        tuple: A tuple containing the cost and accuracy on the validation and test data.
    """
    if initializer == "large":
        net.large_weight_initializer()

    print(f"Training the network using {initializer or 'default'} weight initialization.")

    return net.sgd(
        training_data,
        epochs,
        mini_batch_size,
        eta,
        lmbda=lmbda,
        evaluation_data=validation_data,
        monitor_evaluation_accuracy=True,
    )


def run_network(filename: str, n: int, eta: float):
    """
    Train the network using both the default and the large starting weights.
    Store the results in the file with name `filename`, where they can later be
    used by `make_plots`.

    Args:
        filename (str): Filename to store the results.
        n (int): Number of hidden neurons.
        eta (float): Learning rate.

    """
    # Make results more easily reproducible
    random.seed(SEED_VALUE)
    np.random.seed(SEED_VALUE)

    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    net = network2.Network([784, n, 10], cost=network2.CrossEntropyCost)

    # Train with default and large weight initializations
    default_results = train_network(net, training_data, validation_data, eta)
    large_results = train_network(net, training_data, validation_data, eta, "large")

    # Save results to file
    results = {
        WEIGHT_DEFAULT_KEY: default_results,
        WEIGHT_LARGE_KEY: large_results
    }
    with open(filename, "w") as f:
        json.dump(results, f)


def make_plot(filename: str):
    """
    Load the results from the file ``filename``, and generate the corresponding plot.

    Args:
        filename (str): Filename to load the results from.
    """
    try:
        with open(filename, "r") as f:
            results = json.load(f)
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from file '{filename}'.")
        return

    # Validate keys
    if WEIGHT_DEFAULT_KEY not in results:
        raise ValueError(f"No {WEIGHT_DEFAULT_KEY} in results.")
    if WEIGHT_LARGE_KEY not in results:
        raise ValueError(f"No {WEIGHT_LARGE_KEY} in results.")

    # Extract results
    default_vc, default_va, default_tc, default_ta = results[WEIGHT_DEFAULT_KEY]
    large_vc, large_va, large_tc, large_ta = results[WEIGHT_LARGE_KEY]

    # Convert raw classification numbers to percentages
    default_va = [x / 100.0 for x in default_va]
    large_va = [x / 100.0 for x in large_va]

    # Plot results
    fig, ax = plt.subplots()
    epochs = np.arange(NUM_EPOCHS)
    ax.plot(epochs, large_va, color='#2A6EA6', label="Old approach to weight initialization")
    ax.plot(epochs, default_va, color='#FFA933', label="New approach to weight initialization")

    ax.set_xlim([0, NUM_EPOCHS])
    ax.set_xlabel('Epoch')
    ax.set_ylim([85, 100])
    ax.set_title('Classification accuracy (%)')
    ax.legend(loc="lower right")
    ax.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main("results.json", n=30, eta=0.1)
