import numpy as np
import random, copy
import matplotlib.pyplot as plt

random.seed(1)  #设置随机种子，保证算法结果可以重现

class GeneticAlgorithm:
    def __init__(self, popSize, maxGen, pc, pm, x_range, code_length):
        self.population = []
        self.popSize = popSize
        self.maxGen = maxGen
        self.pc = pc
        self.pm = pm
        self.x_range = x_range
        self.code_length = code_length
        self.history_best = None
        self.history_best_list = []
        self.each_iter_best_list = []


    def run(self):
        iteration = 1
        self.initialize_population()

        while iteration <= self.maxGen:
            temp = self.get_best_individual(self.population)
            self.get_history_best_individual(self.population)
            self.each_iter_best_list.append(temp.fitness)
            self.history_best_list.append(self.history_best.fitness)
            print('Generation: %i | each iteration best: %.6f | history best: %.6f' %
                  (iteration, temp.fitness, self.history_best.fitness))

            selectionResult = self.tournament_selection_Operator(self.population)
            self.crossover_operator(selectionResult)
            self.mutant_operator(selectionResult)
            self.calculate_population_fitness(selectionResult)
            self.population = selectionResult

            iteration += 1

        plt.plot(np.arange(self.maxGen), self.each_iter_best_list)
        plt.plot(np.arange(self.maxGen), self.history_best_list)
        plt.show()


    def get_best_individual(self, population):
        temp = population[0]
        for ind in population:
            if ind.fitness > temp.fitness:
                temp = copy.deepcopy(ind)
        return temp


    def get_history_best_individual(self, population):
        temp = self.get_best_individual(population)
        if self.history_best == None:
            self.history_best = temp
        else:
            if temp.fitness > self.history_best.fitness:
                self.history_best = temp


    def initialize_population(self):
        for i in range(self.popSize):
            ind = Individual()
            ind.chromosome = [random.randint(0,1) for i in range(self.code_length)]
            ind.fitness = self.calculate_individual_fitness(ind)
            self.population.append(ind)


    def tournament_selection_Operator(self, population):
        selectionResult = []
        N = self.popSize
        while (N != 0):
            ind_1 = 0
            ind_2 = 0
            while (ind_1 == ind_2):
                ind_1 = random.randint(0, self.popSize - 1)
                ind_2 = random.randint(0, self.popSize - 1)
            if population[ind_1].fitness > population[ind_2].fitness:
                selectionResult.append(copy.deepcopy(population[ind_1]))
            else:
                selectionResult.append(copy.deepcopy(population[ind_2]))
            N -= 1
        return selectionResult


    def crossover_operator(self, population):
        random.shuffle(population)  # 打乱种群
        for i in range(0, len(population), 2):  # 每隔两个个体取一次
            ind_1 = population[i]
            ind_2 = population[i + 1]
            if random.random() <= self.pc:
                cpt = random.randint(0, self.code_length - 1)  # 单点交叉
                for i in range(cpt):
                    ind_1.chromosome[i], ind_2.chromosome[i] = ind_2.chromosome[i], ind_1.chromosome[i]


    def mutant_operator(self, population):
        for ind in population:
            for i in range(self.code_length):
                if random.random() <= self.pm:
                    ind.chromosome[i] = 1 - ind.chromosome[i]


    def calculate_individual_fitness(self, ind):
        t = 0
        for i in range(len(ind.chromosome)):
            t += ind.chromosome[i] * 2 ** i
        t = self.x_range[0] + t * (self.x_range[1] - self.x_range[0]) / (2 ** self.code_length - 1)


        return 2 * (t**2)


    def calculate_population_fitness(self, population):
        for ind in population:
            ind.fitness = self.calculate_individual_fitness(ind)


class Individual:
    def __init__(self):
        self.chromosome = []      #基因位是SMD类型
        self.fitness = []


if __name__ =='__main__':
    popSize = 20
    maxGen = 100
    pc = 0.7
    pm = 0.01
    x_range = [0, 4]
    code_length = 20

    ga = GeneticAlgorithm(popSize, maxGen, pc, pm, x_range, code_length)
    ga.run()