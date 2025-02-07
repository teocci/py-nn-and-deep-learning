"""
false_minimum
~~~~~~~~~~~~~

Plots a function of two variables with many false minima."""

# Note that axes3d is not explicitly used in the code, but is needed
# to register the 3d plot type correctly
import numpy
import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib.ticker import LinearLocator


def main():
    X = numpy.arange(-5, 5, 0.1)
    Y = numpy.arange(-5, 5, 0.1)
    X, Y = numpy.meshgrid(X, Y)
    Z = numpy.sin(X) * numpy.sin(Y) + 0.2 * X

    colortuple = ('w', 'b')
    colors = numpy.empty(X.shape, dtype=str)
    for x in range(len(X)):
        for y in range(len(Y)):
            colors[x, y] = colortuple[(x + y) % 2]


    # Plot the surface
    cmap = mpl.colormaps['plasma']
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(X, Y, Z, vmin=Z.min() * 2, cmap=cmap)

    # surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=colors, linewidth=0)

    ax.set_xlim3d(-5, 5)
    ax.set_ylim3d(-5, 5)
    ax.set_zlim3d(-2, 2)
    ax.xaxis.set_major_locator(LinearLocator(3))
    ax.yaxis.set_major_locator(LinearLocator(3))
    ax.zaxis.set_major_locator(LinearLocator(3))

    plt.show()


if __name__ == "__main__":
    main()
