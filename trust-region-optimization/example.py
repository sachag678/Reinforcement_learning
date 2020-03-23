import numpy as np
from plot import plot3d


def f(x1, x2):
    return -(x1 ** 2) - 4 * x2 ** 2 + 8 * x1 + 16 * x2


def df(x1, x2):
    return -2 * x1 + 8, -8 * x2 + 16


def dfsingle(t, x1, x2, x1prime, x2prime):
    return (
        -2 * x1prime * (x1 + x1prime * t)
        - 8 * x2prime * (x2 + x2prime * t)
        + 8 * x1prime
        + 16 * x2prime
    )


start = (0, 0)


def ft(f, x1, x2, t):
    return f(x1 * t, x2 * t)


def dft(f, x1, x2, t):
    return f(t, x1, x2)


def bisection_search(df, x1, x2, x1prime, x2prime):

    # find initial points
    step = 0.0001
    xl = 0.0
    while True:
        grad = df(xl, x1, x2, x1prime, x2prime)
        if grad > 0:
            break
        xl = xl + step

    xr = xl + step
    while True:
        grad = df(xr, x1, x2, x1prime, x2prime)
        if grad < 0:
            break
        xr = xr + step

    print("xl: {:.4}, xr: {:.4}".format(xl, xr))

    # iterate
    x = (xl + xr) / 2.0
    for i in range(10000):
        if df(x, x1, x2, x1prime, x2prime) > 0:
            xl = x
        else:
            xr = x
        if (xr - xl) <= 0.001:
            break
        x = (xl + xr) / 2.0
    return x

# initial
points = []
points.append(start)

print("start: {}, f(start): {}".format(start, f(*start)))
grad_x = df(*start)
t = bisection_search(dfsingle, *start, *grad_x)
new = (start[0] + t * grad_x[0], start[1] + t * grad_x[1])
print(
    "grad: ({}, {}) | t: {:.4}, new: ({:.4}, {:.4}), f(new): {:.4}".format(
        *grad_x, t, *new, f(*new)
    )
)

# iterate
for i in range(10):
    points.append(new)
    grad_x = df(*new)
    t = bisection_search(dfsingle, *new, *grad_x)
    new = (new[0] + t * grad_x[0], new[1] + t * grad_x[1])
    print(
        "grad: ({:.4}, {:.4}) | t: {:.4}, new: ({:.4}, {:.4}), f(new): {:.4}".format(
            *grad_x, t, *new, f(*new)
        )
    )

# plot
xs = []
ys = []
for point in points:
    xs.append(point[0])
    ys.append(point[1])

#import matplotlib.pyplot as plt

#plt.plot(xs, ys, "ro--" )
#plt.show()

plot3d(f, points)
