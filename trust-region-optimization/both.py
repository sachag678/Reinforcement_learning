from tro import iterate, cauchy_point, dogleg
from linesearch import findMaxima
import matplotlib.pyplot as plt


def Circle(x, y, r, color):
    circle = plt.Circle((x, y), radius=r, fill=False, edgecolor=color, linestyle="--", alpha=0.5)
    ax.add_patch(circle)

if __name__ == "__main__":

    ax = plt.gca()

    points, deltas = iterate(cauchy_point, 10)

    xs = []
    ys = []

    for point, delta, in zip(points, deltas):
        xs.append(point[0])
        ys.append(point[1])
        Circle(float(point[0]), float(point[1]), delta, 'green')


    points, deltas = iterate(dogleg, 29)
    xsc = []
    ysc = []

    for point, delta, in zip(points, deltas):
        xsc.append(point[0])
        ysc.append(point[1])
        Circle(float(point[0]), float(point[1]), delta, 'cyan')

    lins = findMaxima(10)

    linx = []
    liny = []

    for lin in lins:
        linx.append(lin[0])
        liny.append(lin[1])


    ax.plot(xs, ys, "ro--", linx, liny, "bo--", xsc, ysc, "mo--")
    plt.axis('scaled')
    plt.legend(['TR - Cauchy Point', 'Line Search', 'TR - DogLeg'])
    plt.show()

