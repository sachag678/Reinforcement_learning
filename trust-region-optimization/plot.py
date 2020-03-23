import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# Axes3D import has side effects, it enables using projection='3d' in add_subplot
import matplotlib.pyplot as plt
import random


def plot3d(fun, datas):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    x = y = np.arange(-4.0, 8.0, 0.05)
    X, Y = np.meshgrid(x, y)
    zs = np.array(fun(np.ravel(X), np.ravel(Y)))
    Z = zs.reshape(X.shape)

    surf = ax.plot_surface(X, Y, Z, cmap=cm.terrain)

    ax.set_xlabel("X Label")
    ax.set_ylabel("Y Label")
    ax.set_zlabel("Z Label")

    fig.colorbar(surf, shrink=0.5, aspect=5)

    ax.hold(True)
    ax.plot([data[0] for data in datas], [data[1] for data in datas], [fun(*data) for data in datas], c='r', marker='o')

    plt.show()


def plot2d():
    def func(t):
        return -((8 * t) ** 2) - 4 * (16 * t) ** 2 + 8 * (8 * t) + 16 * 16 * t

    def der(t):
        return -2 * (64 * t) - 8 * 16 * 16 * t + 8 * 8 + 16 * 16

    def base(t):
        return 0 * t

    x = np.arange(0, 2.5, 0.001)
    y = func(x)
    yprime = der(x)

    plt.plot(x, y, x, yprime, x, base(x))
    plt.show()

