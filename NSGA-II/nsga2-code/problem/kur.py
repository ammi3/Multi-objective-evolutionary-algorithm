import math
import numpy as np


def f1(x):
    s = 0
    for i in range(len(x) - 1):
        s += -10 * np.exp(-0.2 * np.sqrt(x[i] ** 2 + x[i + 1] ** 2))
    return s


def f2(x):
    s = 0
    for i in range(len(x)):
        s += np.abs(x[i]) ** 0.8 + 5 * np.sin(x[i] ** 3)
    return s

def get_kur_params():
    objective_functions = [f1, f2]
    variables_range = [[-5, 5] for i in range(3)]
    return objective_functions, variables_range