import math
import numpy as np

def f1(x):
    return x[0]

def g(x):
    value = 0
    for i in range(1, len(x)):
        value = value + x[i] ** 2 - 10 * np.cos(4 * np.pi * x[i])
    return 1 + 10 * (len(x) - 1) + value

def f2(x):
    g_value = g(x)
    return g_value * (1 - np.sqrt(x[0] / g_value))

def get_zdt4_params():
    objective_functions = [f1, f2]
    variables_range = [[0, 1]]
    for i in range(9):
        variables_range.append([-5, 5])
    return objective_functions, variables_range