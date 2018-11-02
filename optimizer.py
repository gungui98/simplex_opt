import numpy as np
from prettytable import PrettyTable


def verbose_print(x, **argv):
    """
    print current state of solving process along with tableau
    """
    tableau = argv["tableau"]
    num_iters = argv["num_iters"]
    pivrow, pivcol = argv["pivot"]
    phase = argv["phase"]
    basis = argv["basis"]
    solved = argv["solved"]
    if solved:
        print("--------- Iteration solved - Phase {0:d} -------\n".format(phase))
        print("Tableau:")
    elif num_iters == 0:
        print("--------- Initial Tableau - Phase {0:d} ----------\n".format(phase))

    else:
        print("--------- Iteration {0:d}  - Phase {1:d} --------\n".format(num_iters, phase))
        print("Tableau:")
    if phase == 1:
        tableau = tableau[0:-1]

    if num_iters >= 0:
        index = ['J']
        for var_i in range(tableau.shape[1] - 1):
            index += ['x{}'.format(var_i + 1)]
        index += ['Xj']
        str_tableau = PrettyTable()
        str_tableau.field_names = index
        count = 0
        for row in tableau[:-1]:
            str_tableau.add_row(sum([[str('x{}'.format(basis[count] + 1))], list(row)], []))
            count += 1
        str_tableau.add_row(sum([['F(x)'], list(tableau[-1])], []))
        print(str_tableau)
        if not solved:
            print("Pivot Element: T[{0:.0f}, {1:.0f}]".format(pivrow, pivcol))
        print("Basic Variables:", basis)
        print("Current Solution:")
        print("x = ", x)
        print("Current Objective Value:")
        print("f = ", -tableau[-1, -1])


def pivot_col(T, epsilon=1.0E-12):
    """
    find the pivot column
    :param T: Tableu
    :param epsilon: tolerate to determine a number is 0 or not
    :return:
    """
    ma = np.ma.masked_where(T[-1, :-1] >= -epsilon, T[-1, :-1], copy=False)
    if ma.count() == 0:
        return False, np.nan
    return True, np.ma.where(ma == ma.min())[0][0]


def pivot_row(T, piv_col, phase, epsilon=1.0E-12):
    """
    find the pivot row
    :param T: Tableu, a 2D matrix
    :param piv_col: indicate of pivot column
    :param phase: solving phase,must be 1 or 2
    :param epsilon: tolerate to determine a number is 0 or not
    :return: indicate of pivot row
    """
    if phase == 1:
        k = 2
    else:
        k = 1
    ma = np.ma.masked_where(T[:-k, piv_col] <= epsilon, T[:-k, piv_col], copy=False)
    if ma.count() == 0:
        return False, np.nan
    mb = np.ma.masked_where(T[:-k, piv_col] <= epsilon, T[:-k, -1], copy=False)
    q = mb / ma
    min_rows = np.ma.where(q == q.min())[0]
    return True, min_rows[0]


def solve_simplex(Tableu, num_var, basis, maxiter=1000, phase=2, print_func=None,
                  epsilon=1.0E-12, num_iter0=0):
    """
    solve the simplex using tableau

    :param Tableu: a matrix contains constains and objective function
    :param num_var: number of real variables
    :param basis: number of basic variable,equals to number of slack variable
    :param maxiter: maximum iteration
    :param phase: phase 1 or phase 2
    :param print_func:
    :param epsilon: determine the objective value is close enough to zero
    """

    num_iters = num_iter0
    solved = False
    status = 0

    if phase == 1:
        m = Tableu.shape[0] - 2
    elif phase == 2:
        m = Tableu.shape[0] - 1
    else:
        raise ValueError("Argument 'phase' to solve must be 1 or 2")

    if len(basis[:m]) == 0:
        solution = np.zeros(Tableu.shape[1] - 1, dtype=np.float64)
    else:
        solution = np.zeros(max(Tableu.shape[1] - 1, max(basis[:m]) + 1),
                            dtype=np.float64)

    while not solved:
        # Find the pivot column
        is_found_col, piv_col = pivot_col(Tableu, epsilon)
        if not is_found_col:
            piv_col = np.nan
            piv_row = np.nan
            status = 0
            solved = True
        else:
            # Find the pivot row
            is_found_row, piv_row = pivot_row(Tableu, piv_col, phase, epsilon)
            if not is_found_row:
                return num_iters, 3

        if print_func is not None:
            solution[:] = 0
            solution[basis[:m]] = Tableu[:m, -1]
            print_func(solution[:num_var], **{"tableau": Tableu,
                                              "num_var": num_var,
                                              "phase": phase,
                                              "num_iters": num_iters,
                                              "pivot": (piv_row, piv_col),
                                              "basis": basis,
                                              "solved": solved and phase == 2})

        if not solved:
            if num_iters >= maxiter:
                status = 1
                solved = False
            else:
                # variable represented by piv_col enters
                # variable in basis[piv_row] leaves
                basis[piv_row] = piv_col
                pivval = Tableu[piv_row][piv_col]
                Tableu[piv_row, :] = Tableu[piv_row, :] / pivval
                for irow in range(Tableu.shape[0]):
                    if irow != piv_row:
                        Tableu[irow, :] = Tableu[irow, :] - Tableu[piv_row, :] * Tableu[irow, piv_col]
                num_iters += 1

    return num_iters, status


def simplex_problem(c, A=None, b_val=None,
                    maxiter=1000, print_func=None,
                    epsilon=1.0E-12):
    """
    modelize the problem in to slack form and solve it
    with problem like:
    minimize: c.T * x
    given: A.T * x <= b_val

    :param c: coef of objective function
    :param A: coef of constrains
    :param b_val: value of constrain
    :param maxiter: maximum iteration in solve step
    :param print_func: print function
    :param epsilon: determine the value of objective function is closed enough to 0
    :return: dictionary contain result
    """
    messages = {0: "successful.",
                1: "Iteration limit reached.",
                2: "Error. Unable to find feasible region",
                3: "Error. The obj function appears to decease to - inf."}

    c_array = np.asarray(c)

    # The initial value of the objective function element in the tableau
    f0 = 0

    # The number of variables as given by c
    num_of_var = len(c)

    # Convert the input arguments to arrays (sized to zero if not provided)

    A = np.asarray(A) if A is not None else np.empty([0, len(c_array)])
    b_val = np.ravel(np.asarray(b_val)) if b_val is not None else np.empty([0])

    Lower_bound = np.zeros(num_of_var, dtype=np.float64)
    Upper_bound = np.ones(num_of_var, dtype=np.float64) * np.inf

    num_const = len(b_val)
    n_slack = num_const

    try:
        A_row, A_col = A.shape
    except ValueError:
        raise ValueError("A must has two dimensions")

    if A_row != num_const:
        raise ValueError("The number of rows in A must be equal to the number of values in b_val")

    if A_col > 0 and A_col != num_of_var:
        raise ValueError("Number of columns in A must be equal to the size of c")

    # Create the tableau
    Tableu = np.zeros([num_const + 2, num_of_var + n_slack + 1])
    # Insert objective into tableau
    Tableu[-2, :num_of_var] = c_array
    # Initialize equals to 0
    Tableu[-2, -1] = 0

    b = Tableu[:-2, -1]

    if num_const > 0:
        # Add A to the tableau
        Tableu[0:num_const, :num_of_var] = A
        # At b_val to the tableau
        b[0:num_const] = b_val
        # Add the slack variables to the tableau
        np.fill_diagonal(Tableu[0:num_const, num_of_var:num_of_var + n_slack], 1)

    # Add slack variable to tableau
    num_slack = 0
    basis = np.zeros(num_const, dtype=int)
    for i in range(num_const):
        # basic variable i is in column n+num_slack
        basis[i] = num_of_var + num_slack
        num_slack += 1

    nit1, status = solve_simplex(Tableu, num_of_var, basis, phase=1, print_func=print_func,
                                 maxiter=maxiter, epsilon=epsilon)

    # Phase 2
    if abs(Tableu[-1, -1]) < epsilon:
        # Remove the pseudo-objective row from the tableau
        Tableu = Tableu[:-1, :]
    else:
        # Failure to find a feasible starting point
        raise ValueError("Unable to find a feasible")

    # Phase 2
    nit2, status = solve_simplex(Tableu, num_of_var, basis, maxiter=maxiter - nit1, phase=2,
                                 print_func=print_func, epsilon=epsilon, num_iter0=nit1)

    solution = np.zeros(num_of_var + n_slack)
    solution[basis[:num_const]] = Tableu[:num_const, -1]
    x = solution[:num_of_var]
    slack = solution[num_of_var:num_of_var + n_slack]

    obj = -Tableu[-1, -1]

    return {'x': x, 'obj value': obj, 'status': status,
            'slack': slack, 'messages': messages[status],
            'success': (status == 0)}
