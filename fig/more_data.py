"""more_data
~~~~~~~~~~~~

Plot graphs to illustrate the performance of MNIST when different size
training sets are used.

"""

# Standard library
import json
import random

# Third-party libraries
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm

# My library
import src.mnist_loader_e as mnist_loader
import src.network2_e as network2
from fig.weight_initialization_e import LAMBDA

# Constants
SEED_VALUE = 12345678
# The sizes to use for the different training sets
SIZES = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]
EPOCH_MULTIPLIER = 1500000
LAMBDA_MULTIPLIER = 0.0001
MIN_BATCH_SIZE = 10
ETA = 0.5

def run_networks():
    # Make results more easily reproducible
    random.seed(SEED_VALUE)
    np.random.seed(SEED_VALUE)
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost())
    accuracies = []
    for size in SIZES:
        num_epochs = int(EPOCH_MULTIPLIER / size)
        lmbda = size * LAMBDA_MULTIPLIER
        print(f"\n\nTraining network with data set size {size}")
        net.large_weight_initializer()
        net.sgd(training_data[:size], num_epochs, MIN_BATCH_SIZE, ETA, lmbda=lmbda)
        accuracy = net.accuracy(validation_data) / 100.0
        print(f"Accuracy was {accuracy} percent")
        accuracies.append(accuracy)
    with open("more_data.json", "w") as f:
        json.dump(accuracies, f)


def run_svms():
    svm_training_data, svm_validation_data, svm_test_data \
        = mnist_loader.load_data()
    accuracies = []
    for size in SIZES:
        print(f"\n\nTraining SVM with data set size {size}")
        clf = svm.SVC()
        clf.fit(svm_training_data[0][:size], svm_training_data[1][:size])
        # predictions = [int(a) for a in clf.predict(svm_validation_data[0])]
        # accuracy = sum(int(a == y) for a, y in
        #                zip(predictions, svm_validation_data[1])) / 100.0
        predictions = clf.predict(svm_validation_data[0])
        accuracy = np.mean(predictions == svm_validation_data[1]) * 100
        print(f"Accuracy was {accuracy} percent")
        accuracies.append(accuracy)
    with open("more_data_svm.json", "w") as f:
        json.dump(accuracies, f)


def make_plot(accuracies, svm_accuracies=None, log_scale=False, combined=False):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(SIZES, accuracies, color='#2A6EA6', label='Neural network accuracy (%)' if combined else None)
    ax.plot(SIZES, accuracies, "o", color='#FFA933')
    if combined:
        ax.plot(SIZES, svm_accuracies, color='#FFA933', label='SVM accuracy (%)')
        ax.plot(SIZES, svm_accuracies, "o", color='#FFA933')
        ax.set_ylim(25, 100)
        plt.legend(loc="lower right")
    else:
        ax.set_ylim(60, 100)
    ax.set_xlim(100 if log_scale else 0, 50000)
    if log_scale:
        ax.set_xscale('log')
    ax.grid(True)
    ax.set_xlabel('Training set size')
    ax.set_title('Accuracy (%) on the validation data')
    plt.show()


def make_plots():
    with open("more_data.json", "r") as f:
        accuracies = json.load(f)
    with open("more_data_svm.json", "r") as f:
        svm_accuracies = json.load(f)
    make_plot(accuracies)
    make_plot(accuracies, log_scale=True)
    make_plot(accuracies, svm_accuracies, log_scale=True, combined=True)
    # make_linear_plot(accuracies)
    # make_log_plot(accuracies)
    # make_combined_plot(accuracies, svm_accuracies)


def make_linear_plot(accuracies):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(SIZES, accuracies, color='#2A6EA6')
    ax.plot(SIZES, accuracies, "o", color='#FFA933')
    ax.set_xlim(0, 50000)
    ax.set_ylim(60, 100)
    ax.grid(True)
    ax.set_xlabel('Training set size')
    ax.set_title('Accuracy (%) on the validation data')
    plt.show()


def make_log_plot(accuracies):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(SIZES, accuracies, color='#2A6EA6')
    ax.plot(SIZES, accuracies, "o", color='#FFA933')
    ax.set_xlim(100, 50000)
    ax.set_ylim(60, 100)
    ax.set_xscale('log')
    ax.grid(True)
    ax.set_xlabel('Training set size')
    ax.set_title('Accuracy (%) on the validation data')
    plt.show()


def make_combined_plot(accuracies, svm_accuracies):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(SIZES, accuracies, color='#2A6EA6')
    ax.plot(SIZES, accuracies, "o", color='#2A6EA6', label='Neural network accuracy (%)')
    ax.plot(SIZES, svm_accuracies, color='#FFA933')
    ax.plot(SIZES, svm_accuracies, "o", color='#FFA933', label='SVM accuracy (%)')
    ax.set_xlim(100, 50000)
    ax.set_ylim(25, 100)
    ax.set_xscale('log')
    ax.grid(True)
    ax.set_xlabel('Training set size')
    plt.legend(loc="lower right")
    plt.show()


def main():
    run_networks()
    run_svms()
    make_plots()


if __name__ == "__main__":
    main()
