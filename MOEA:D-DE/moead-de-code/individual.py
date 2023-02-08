import numpy as np

class Individual:
    def __init__(self):
        self.genes = None
        self.fitness = None

    # 获取适应值
    def get_fitness(self, objective_functions):
        len_funcs = len(objective_functions)
        # print(len_funcs)
        fitness = np.zeros(shape=len_funcs)
        for i in range(len_funcs):
            fitness[i] = objective_functions[i](self.genes)
        return fitness

    def dominate(self, ind):
        flag = -1
        for i in range(len(self.fitness)):
            if self.fitness[i] < ind.fitness[i]:
                flag = 0
            elif self.fitness[i] > ind.fitness[i]:
                return False
        if flag == 0:
            return True
        else:
            return False
