import numpy as np
from matplotlib import pyplot as plt
from optimizer import simplex_problem, verbose_print

import seaborn as sns

sns.set_palette('Set1')

paths = []


def feasible_region(A, b):
    """
    calculate the feasible region of LP prblem
    :param A: coef of constrains
    :param b: upper bound of constrains
    :return: 2D coordinate of point in feasible region
    """
    b = np.expand_dims(b, axis=1)
    x0 = np.linspace(0, 100, 500)
    x = np.array(np.meshgrid(x0, x0))
    # Construct grid
    x = x.reshape(2, -1)
    checked_cons = np.matmul(A, x) <= np.repeat(b, x.shape[1], axis=1)
    # and logic product in column
    mask = checked_cons.all(axis=0).reshape((x0.shape[0], x0.shape[0]))
    x = x.reshape((2, x0.shape[0], x0.shape[0]))
    x = (x.T[mask.T]).T
    return x


def visual_problem(c, A, b, optimal=None):
    """
    visualize for 2 variable constrains and feasible region for LP problem

    Given A.T * x <= b,that mean
    A[0]*x0 + A[1]*x1 <= b[0]
    Minimize
    c.T * x
    :param optimal: optimal solution
    :param A: constrain coefs
    :param b: constrain value
    :return: plt figure
    """

    if c.shape[0] != 2:
        raise Exception('function must be 2 dimension')

    epsilon = 0.000000001

    if A.shape[0] != b.shape[0]:
        raise Exception('A and b must be the same shape')

    fig, ax = plt.subplots(figsize=(8, 8))
    x1 = np.linspace(0, 100)

    for i in range(A.shape[0]):
        plt.plot(x1, (b[i] - A[i][0] * x1) / (A[i][1] + epsilon), lw=3,
                 label=r'{} : ${}x_1 + {}x_2 \leq{}$'.format(i, A[i][0], A[i][1], b[i]))
        plt.fill_between(x1, 0, (b[i] - A[i][0] * x1) / (A[i][1] + epsilon), alpha=0.1)

    plt.plot(np.zeros_like(x1), x1, lw=3, label=r'$x_1 : non-negative$')
    plt.plot(x1, np.zeros_like(x1), lw=3, label=r'$x_2 : non-negative$')
    # calculate the feasible
    fx, fy = feasible_region(A, b)
    z = np.dot(c.T, [fx, fy])
    plt.scatter(fx, fy, c=z, cmap='jet', zorder=-1)
    if optimal is not None:
        plt.plot(optimal[0], optimal[1], 'ro', label='optimal point')
    # labels and stuff
    cb = plt.colorbar()
    cb.set_label('objective value', fontsize=14)
    plt.xlabel('x1', fontsize=16)
    plt.ylabel('x2', fontsize=16)
    plt.xlim(-0.5, 100)
    plt.ylim(-0.5, 100)
    plt.legend(fontsize=14)
    plt.show()


if __name__ == '__main__':
    c = np.array([-8, -2])
    A = np.array([[1, 1],
                  [2, 1],
                  [1, 0]])
    b = np.array([80, 100, 40])
    res = simplex_problem(c, A, b, print_func=verbose_print)
    visual_problem(c, A, b, res['x'])
