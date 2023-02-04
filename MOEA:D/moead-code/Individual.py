import numpy as np


class Individual:
    def __init__(self):
        self.genes = None
        self.genes_size = None
        self.rank = None
        self.fitness = None
        self.weight_vector = None
        self.neighbors = None
        self.objective_functions = None

    def dominate(self, individual2):
        individual1 = self
        first_condition = True
        second_condition = False
        for i in range(len(individual1.fitness)):
            if individual1.fitness[i] > individual2.fitness[i]:
                first_condition = False
            elif individual1.fitness[i] < individual2.fitness[i]:
                second_condition = True
        return first_condition and second_condition

    # 获取适应值
    def get_fitness(self, objective_functions):
        len_funcs = len(objective_functions)
        # print(len_funcs)
        fitness = np.zeros(shape=len_funcs)
        for i in range(len_funcs):
            fitness[i] = objective_functions[i](self.genes)
        return fitness




