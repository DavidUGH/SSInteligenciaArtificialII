import math
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def f(x, y):
    return (10 - math.exp(1)**(-((x**2)+(3*(y**2)))))

#def grad(x):
#    return 8*(math.exp(1)**(-4**(x**2)))*x

def grad(x,y):
    g_x = -(2*x*(math.exp(1)**(-((x**2)+(3*x**2)))))*x
    g_y = -(6*y*(math.exp(1)**(-((x**2)+(3*y**2)))))*y
    return g_x, g_y

def grad_des(x1, x2, lr, it):
    x = x1
    y = x2
    path = []
    convergence = []
    path.append((x,y,f(x,y)))
    convergence.append(f(x,y))

    for i in range(it):
        g_x = x * 2
        g_y = y * 2
        x = x - lr * g_x
        y = y - lr * g_y
        path.append((x,y,f(x,y)))
        convergence.append(f(x,y))
    return path, convergence, x, y

def main():
    lr = float(input("Ingrese el learning rate "))
    it = int(input("Ingrese el numero de iteraciones "))

    x_dots = np.arange(-1, 1, 0.2)
    y_dots = np.arange(-1, 1, 0.2)
    x_grid, y_grid = np.meshgrid(x_dots, y_dots)
    z_grid = f(x_grid, y_grid)

    x0 = random.uniform(-1.0, 1.0)
    y0 = random.uniform(-1.0, 1.0)

    path, convergence, last_x, last_y = grad_des(x0, y0, lr, it)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(x_grid, y_grid, z_grid)
    ax.scatter(*zip(*path), c='red', marker='*', s=50, alpha=0.6)
    label = "X:" + str(last_x) + ", Y:" + str(last_y)
    plt.title("Minimum\n"+label)

    fig2 = plt.figure()
    ax = fig2.add_subplot(1, 1, 1)
    ax.plot(convergence)
    ax.set_title('Convergence Graph')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Objective Value')

    plt.tight_layout()
    plt.show()

main()

