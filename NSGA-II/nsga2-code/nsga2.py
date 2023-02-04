import random

import numpy as np
from individual import Individual

class NSGA2:
    def __init__(self, population_size, genes_size, crossover_prob, mutation_prob, objective_functions, variables_range):
        self.population_size = population_size
        self.genes_size = genes_size
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.objective_functions = objective_functions
        self.objective_functions_len = len(objective_functions)
        self.variables_range = variables_range

    # 生成个体
    def generate_individual(self, genes):
        individual = Individual()
        individual.genes = genes
        individual.genes_len = len(genes)
        return individual

    def get_genes(self, individuals_size, genes_size):
        individuals_genes = np.random.random(size=(individuals_size, genes_size))
        for i in range(len(self.variables_range)):
            variable_range = self.variables_range[i]
            individuals_genes[:, i] = individuals_genes[:, i] * (variable_range[1] - variable_range[0]) + variable_range[0]
        return individuals_genes

    # 构造初始化种群
    def generate_initial_population(self):
        # population_genes = np.random.random(size=(self.population_size, self.genes_size))
        # for i in range(len(self.variables_range)):
        #     variable_range = self.variables_range[i]
        #     population_genes[:, i] = population_genes[:, i] * (variable_range[1] - variable_range[0]) + variable_range[0]

        population_genes = self.get_genes(self.population_size, self.genes_size)
        population = []
        for genes in population_genes:
            population.append(self.generate_individual(genes))

        return population

    # 计算种群中个体的适应度值
    def calculate_objectives(self, population):
        for individual in population:
            individual.fitness = individual.get_fitness(self.objective_functions)
            # print(individual.fitness)
        return population

    # 快速非支配排序
    def fast_nondominated_sort(self, population):
        population_size = len(population)
        S = []   # 存储每个个体所支配的个体
        n = np.zeros(shape=population_size)   # 存储每个个体的被支配数
        fronts_idx = [[]]
        for i in range(population_size):
            S_p = []
            n[i] = 0
            for j in range(population_size):
                if i == j:
                    continue
                if population[i].dominate(population[j]):
                    S_p.append(j)
                elif population[j].dominate(population[i]):
                    n[i] += 1
            S.append(S_p)
            if n[i] == 0:
                population[i].rank = 0
                fronts_idx[0].append(i)

        i = 0
        while len(fronts_idx[i]) > 0:
            Q = []
            for p in fronts_idx[i]:
                for q in S[p]:
                    n[q] -= 1
                    if n[q] == 0:
                        population[q].rank = i + 1
                        Q.append(q)
            i += 1
            fronts_idx.append(Q)
        fronts_idx.pop()

        fronts = []
        for rank in fronts_idx:
            rank_individuals = [population[idx] for idx in rank]
            fronts.append(rank_individuals)
        return fronts

    # 计算拥挤距离
    def calculate_crowding_distance(self, fronts):
        for front in fronts:
            front_fitness = []
            for individual in front:
                individual.crowding_distance = 0
                front_fitness.append(individual.fitness)
            front_fitness = np.array(front_fitness)
            fitness_sorted = np.argsort(front_fitness, axis=0)
            functions_len = len(self.objective_functions)
            for i in range(functions_len):
                fitness_sorted_by_i = fitness_sorted[:, i]
                sorted_population = [front[i] for i in fitness_sorted_by_i]
                sorted_population[0].crowding_distance = sorted_population[-1].crowding_distance = float('inf')
                diff = sorted_population[-1].fitness[i] - sorted_population[0].fitness[i] + 1
                for j in range(1, len(sorted_population) - 1):
                    sorted_population[j].crowding_distance += (sorted_population[j + 1].fitness[i] - sorted_population[j - 1].fitness[i]) / diff

            # print(front)

    # 拥挤比较算子
    def crowding_operator(self, individual, other_individual):
        if (individual.rank < other_individual.rank) or \
                ((individual.rank == other_individual.rank) and (
                        individual.crowding_distance > other_individual.crowding_distance)):
            return 1
        else:
            return -1

    # 遗传操作
    def generate_children(self, population):
        children = []
        while len(children) < len(population):
            parent1 = self.tournament_selection(population)
            parent2 = parent1;
            while parent1 == parent2:
                parent2 = self.tournament_selection(population)
            child1, child2 = self.crossover(parent1, parent2)
            self.mutation(child1, self.variables_range)
            self.mutation(child2, self.variables_range)
            # for function in self.objective_functions:
            #     print(function)
            child1.fitness = child1.get_fitness(self.objective_functions)
            child2.fitness = child2.get_fitness(self.objective_functions)

            children.append(child1)
            children.append(child2)
        return children

    # 锦标赛选择
    def tournament_selection(self, population):
        participants = random.sample(population, 2)
        best = None
        for participant in participants:
            if best is None or (self.crowding_operator(participant, best) == 1):
                best = participant
        return best

    # 交叉操作
    def crossover(self, parent1, parent2):
        child1_genes = self.get_genes(1, self.genes_size)
        child2_genes = self.get_genes(1, self.genes_size)
        child1 = self.generate_individual(child1_genes[0])
        child2 = self.generate_individual(child2_genes[0])
        genes_len = child1.genes_len
        crossover_idx = random.randint(0, genes_len)
        for i in range(0, genes_len):
            if i < crossover_idx:
                child1.genes[i] = parent2.genes[i]
                child2.genes[i] = parent1.genes[i]
            else:
                child1.genes[i] = parent1.genes[i]
                child2.genes[i] = parent2.genes[i]
        return child1, child2

    # 变异操作
    def mutation(self, individual, variables_range):
        if np.random.rand() < self.mutation_prob:
            genes_len = individual.genes_len
            mutation_idx = np.random.randint(0, genes_len)
            minVal = variables_range[mutation_idx][0]
            maxVal = variables_range[mutation_idx][1]
            individual.genes[mutation_idx] = np.random.rand() * (maxVal - minVal) + minVal
