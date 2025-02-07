"""
misleading_gradient
~~~~~~~~~~~~~~~~~~~

Plots a function which misleads the gradient descent algorithm."""
# Note that axes3d is not explicitly used in the code, but is needed
# to register the 3d plot type correctly

import matplotlib.pyplot as plt
import numpy

from matplotlib.ticker import LinearLocator


def main():
    X = numpy.arange(-1, 1, 0.025)
    xlen = len(X)
    Y = numpy.arange(-1, 1, 0.025)
    ylen = len(Y)
    X, Y = numpy.meshgrid(X, Y)
    Z = X ** 2 + 10 * Y ** 2

    colortuple = ('w', 'w', 'b', 'b')
    colors = numpy.empty(X.shape, dtype=str)
    for x in range(xlen):
        for y in range(ylen):
            v = (x + y) % len(colortuple)
            colors[x, y] = colortuple[v]

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(X, Y, Z, vmin=Z.min() * 2, facecolors=colors, linewidth=0)

    ax.set_xlim3d(-1, 1)
    ax.set_ylim3d(-1, 1)
    ax.set_zlim3d(0, 12)
    ax.xaxis.set_major_locator(LinearLocator(3))
    ax.yaxis.set_major_locator(LinearLocator(3))
    ax.zaxis.set_major_locator(LinearLocator(3))
    ax.text(0.05, -1.8, 0, "$w_1$", fontsize=20)
    ax.text(1.5, -0.25, 0, "$w_2$", fontsize=20)
    ax.text(1.79, 0, 9.62, "$C$", fontsize=20)

    plt.show()


if __name__ == "__main__":
    main()
