import numpy as np

from optimizer import simplex_problem, verbose_print

if __name__ == '__main__':
    # c = np.array([-8, -2])
    # A = np.array([[1, 1],
    #               [2, 1],
    #               [1, 0]])
    # b = np.array([80, 100, 40])

    # c = np.array([-8, -6])
    # A = np.array([[4, 2],
    #               [2, 4]])
    # b = np.array([60, 48])

    # c = np.array([2, -2, 1, -2])
    # A = np.array([[2, 5, -3, 1],
    #               [-8, 0, 1, -4],
    #               [3, -4, 2, 1],
    #               [-3, 4, -2, -1]])
    # b = np.array([-46, 38, 24, -24])

    c = np.array([1, -4, -3])
    A = np.array([[2, 1, -2],
                  [-4, 0, 2],
                  [1, 2, -3]])
    b = np.array([16, 28, 12])

    res = simplex_problem(c, A, b, print_func=verbose_print)

    print(res)
