"""
Created by Teocci.
Author: teocci@yandex.com
Date: 2025-2월-07
"""

# Standard library
import json
import random
import sys
from pathlib import Path

# My library
sys.path.append('../src/')
from src import network2_e as network2, mnist_loader

# Third-party libraries
import matplotlib.pyplot as plt
import numpy as np
from concurrent.futures import ThreadPoolExecutor  # Optional for parallelization

# Constants
LEARNING_RATES = [0.025, 0.25, 2.5]
COLORS = ['#2A6EA6', '#FFCD33', '#FF7033']
NUM_EPOCHS = 30
RESULTS_FILE = Path("multiple_eta.json")


def main():
    run_networks()
    make_plot()


def run_networks():
    """Train networks using three different values for the learning rate,
    and store the cost curves in the file ``multiple_eta.json``, where
    they can later be used by ``make_plot``.
    """
    # Make results more easily reproducible
    random.seed(12345678)
    np.random.seed(12345678)
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    results = []

    def train_network(eta):
        print(f"Training network using eta = {eta}")
        net = network2.Network([784, 30, 10])
        return net.sgd(
            training_data, NUM_EPOCHS, 10, eta, lmbda=5.0,
            evaluation_data=validation_data,
            monitor_training_cost=True
        )

    # Optional: Parallelize training using ThreadPoolExecutor
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(train_network, LEARNING_RATES))

    # Save results to file
    with RESULTS_FILE.open("w") as f:
        json.dump(results, f)


def make_plot():
    """Plot the training cost curves for different learning rates."""
    with RESULTS_FILE.open("r") as f:
        results = json.load(f)

    fig, ax = plt.subplots()
    for eta, result, color in zip(LEARNING_RATES, results, COLORS):
        _, _, training_cost, _ = result
        ax.plot(
            np.arange(NUM_EPOCHS),
            training_cost,
            "o-",
            label=f"$\eta$ = {eta}",
            color=color
        )
    ax.set_xlim([0, NUM_EPOCHS])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Cost')
    ax.legend(loc='upper right')
    ax.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
