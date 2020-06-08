import numpy as np
import scipy.linalg as al
import math
from linesearch import f, df, hessian
from plot import plot3d

def model(x1, x2, p1=0, p2=0):
    g = df(x1, x2)
    B = hessian()
    return f(x1, x2) + g[0]*p1 + g[1]*p2 + 0.5*(B[0][0]*p1**2 + B[1][1]*p2**2)

def func(x1, x2, p1=0, p2=0):
    return f(x1+p1, x2+p2)

def cauchy_point(x1, x2, delta, returnVec=False):
    g = df(x1, x2)
    gvec = np.array([[g[0], g[1]]])
    B = hessian()
    check = (gvec @ B @ gvec.transpose()).flatten()[0]
    print('cauchy_check for tau: {}'.format(check))
    if check <= 0:
        tau = 1
    else:
        tau = min(al.norm(g)**3/(delta*check), 1)
    factor = (tau*delta/al.norm(g))

    if returnVec:
        return np.array([factor*g[0], factor*g[1]])
    else:
        return factor*g[0], factor*g[1]

def dogleg(x1, x2, delta):
    def quadratic(t, a, b, c):
        return a*t**2 + b*t + c

    def quadprime(t, a, b):
        return 2*a*t + b

    def newton_raphson(a, b, c):
        t = 0.1
        for i in range(100):
            told = t
            t = t - (quadratic(t, a, b, c)/quadprime(t, a, b))
            if (abs(told - t)) < 0.0001 :
                break
            print(abs(told - t))
        return t

    def pb(x1, x2):
        g = df(x1, x2)
        B = hessian()
        return (al.inv(B) @ np.array([[g[0], g[1]]]).transpose()).transpose().flatten()

    def pu(x1, x2, delta):
        return cauchy_point(x1, x2, delta, True)

    def tau(lu, lb, delta):

        # solving ||pu+(t-1)(pb-pu)||**2 = delta**2
        pc = lb - lu
        pa = 2*lu - lb

        # quadratic terms
        a = pc[0]**2+pc[1]**2
        b = 2*(pc[1]*pa[1]+pc[0]*pa[0])
        c = pa[0]**2 + pa[1]**2 - delta**2

        print(a, b, c)

        #solve quadratic to find t
        return newton_raphson(a, b, c)

    def ptau(x1, x2, delta):
        lb = pb(x1, x2)
        lu = pu(x1, x2, delta)

        print(lb)
        print(lu)

        if al.norm(lu) >= delta:
            g = df(x1, x2)
            gvec = np.array([[g[0], g[1]]])
            return ((delta/al.norm(g)**2)*gvec).flatten()

        if al.norm(lb) <= delta:
            return lb

        t = tau(lu, lb, delta)

        if t >= 0 and t <= 1:
            return t*lu
        else:
            return lu + (t - 1)*(lb - lu)

    return ptau(x1, x2, delta)


def iterate(getp, num_iter):
    points = []
    deltas = []

    max_delta = 1.0 # decide this?
    delta = 0.5
    nu = 0.15

    x = (0, 0)
    points.append(x)
    deltas.append(delta)

    for i in range(num_iter):
        #obtain pk
        p = getp(*x, delta)
        print('p: {}'.format(p))

        #obtain rhok
        rho = (func(*x) - func(*x, *p))/(model(*x)- model(*x, *p))
        print('rho: {}'.format(rho))

        print('before delta: {}'.format(delta))
        #handles the trust region
        if rho < 0.25:
            delta = delta*0.25
        else:
            if rho > 0.75 and al.norm(p) == delta:
                delta = min(2*delta, max_delta)
                print('Increase size')
            else:
                delta = delta

        print('after delta: {}'.format(delta))

        print('x: {}'.format(x))
        #handles the update
        if rho > nu:
            x = (x[0] + p[0], x[1] + p[1])
        else:
            x = x

        points.append(x)
        deltas.append(delta)

        print('x: {}'.format(x))

    return points, deltas

def plot(points, deltas):
    import matplotlib.pyplot as plt

    ax = plt.gca()

    def Circle(x, y, r):
        circle = plt.Circle((x, y), radius=r, fill=False)
        ax.add_patch(circle)

    xs = []
    ys = []
    c = []
    for point, delta in zip(points, deltas):
        xs.append(point[0])
        ys.append(point[1])
        Circle(float(point[0]), float(point[1]), delta)

    ax.plot(np.array(xs), np.array(ys), "ro--")
    plt.axis('scaled')
    plt.show()

if __name__ == "__main__":
    points, deltas = iterate(dogleg, 29)
    plot(points, deltas)
