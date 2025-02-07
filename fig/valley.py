"""
valley
~~~~~~

Plots a function of two variables to minimize.  The function is a
fairly generic valley function."""

# Note that axes3d is not explicitly used in the code, but is needed
# to register the 3d plot type correctly
import matplotlib.pyplot as plt
import numpy
#### Libraries
# Third party libraries
from matplotlib.ticker import LinearLocator


def main():
    X = numpy.arange(-1, 1, 0.1)
    xlen = len(X)
    Y = numpy.arange(-1, 1, 0.1)
    ylen = len(Y)
    X, Y = numpy.meshgrid(X, Y)
    Z = X ** 2 + Y ** 2

    colortuple = ('w', 'b')
    colors = numpy.empty(X.shape, dtype=str)
    for x in range(xlen):
        for y in range(ylen):
            colors[x, y] = colortuple[(x + y) % 2]

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(X, Y, Z, vmin=Z.min() * 2, rstride=1, cstride=1, facecolors=colors, linewidth=0)

    ax.set_xlim3d(-1, 1)
    ax.set_ylim3d(-1, 1)
    ax.set_zlim3d(0, 2)
    ax.xaxis.set_major_locator(LinearLocator(3))
    ax.yaxis.set_major_locator(LinearLocator(3))
    ax.zaxis.set_major_locator(LinearLocator(3))
    ax.set_xlabel('$v_1$', fontsize=20)
    ax.set_ylabel('$v_2$', fontsize=20)
    ax.set_zlabel('$C$', fontsize=20)

    plt.show()


if __name__ == "__main__":
    main()
