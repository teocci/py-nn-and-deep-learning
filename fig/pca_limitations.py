"""
pca_limitations
~~~~~~~~~~~~~~~

Plot graphs to illustrate the limitations of PCA.
"""
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

SEED_VALUE = 42


def generate_noisy_helix(n_points=20, noise_level=0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a noisy 3D helix-like dataset.

    Args:
        n_points (int): Number of data points.
        noise_level (float): Standard deviation of Gaussian noise.

    Returns:
        tuple: (x, y, z) NumPy arrays for 3D coordinates.
    """

    def noise(v, n, level=0.0) -> np.ndarray:
        if level == 0:
            return v

        rng = np.random.default_rng(seed=SEED_VALUE)
        return v + level * rng.standard_normal(n)

    theta = np.linspace(-4 * np.pi, 4 * np.pi, n_points)

    z = np.linspace(-2, 2, n_points)
    x = noise(np.sin(theta), n_points, noise_level)
    y = noise(np.cos(theta), n_points, noise_level)
    return x, y, z


def plot_noisy_helix():
    """Plot noisy helix data points."""

    # Generate the noisy data
    x, y, z = generate_noisy_helix(noise_level=0.03)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c='r', marker='o', label="Noisy Data")

    ax.set_title("Noisy Helix Data")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")
    ax.legend()
    plt.show()


def plot_helix_with_true_curve():
    """Plot noisy helix data along with the true helix curve."""

    # Generate the noisy data
    x, y, z = generate_noisy_helix(noise_level=0.03)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Generate the helix data
    x_helix, y_helix, z_helix = generate_noisy_helix(n_points=100)

    ax.scatter(x, y, z, c='r', marker='o', label="Noisy Data")
    ax.plot(x_helix, y_helix, z_helix, 'b-', linewidth=2, label="True Helix")

    ax.set_title("Noisy Data vs True Helix")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")
    ax.legend()
    plt.show()


def main():
    """Main function to plot PCA limitations."""
    plot_noisy_helix()
    plot_helix_with_true_curve()


if __name__ == "__main__":
    main()
