def f1(x):
    return x**2

def f2(x):
    return (x - 2)**2

def get_sch_params():
    objective_functions = [f1, f2]
    variables_range = [[-1000, 1000] for i in range(1)]
    return objective_functions, variables_range