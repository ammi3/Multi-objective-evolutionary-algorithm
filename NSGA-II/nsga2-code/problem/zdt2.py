import math
import numpy as np

def f1(x):
    return x[0]

def g(x):
    return 1 + 9 * (x[1:].sum()) / (len(x) - 1)

def f2(x):
    g_value = g(x)
    return g_value * (1 - (x[0] / g_value) ** 2)

def get_zdt2_params():
    objective_functions = [f1, f2]
    variables_range = [[0, 1] for i in range(30)]
    return objective_functions, variables_range