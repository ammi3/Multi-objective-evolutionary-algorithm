import numpy as np
from Individual import Individual
from utils.weight_vectors_util import generate_weight_vectors

class MOEAD:
    def __init__(self, H, m, T, genes_size, mutation_prob, objective_functions, variables_range):
        self.H = H   # 取点密度
        self.m = m   # 决策函数数目
        self.T = T   # 邻居数

        self.genes_size = genes_size
        self.mutation_prob = mutation_prob
        self.objective_functions = objective_functions
        self.variables_range = variables_range
        self.Z = np.zeros(shape=m)

        self.weight_vectors = None
        self.Euler_distance = None

        self.population = None
        self.population_size = None
        self.EP = []
        self.EP_fitness = []

    def init_weight_vectors(self):
        weight_vectors = generate_weight_vectors(self.H + self.m, self.m)
        weight_vectors = (np.array(weight_vectors) - 1) / self.H
        self.weight_vectors = weight_vectors
        self.population_size = len(weight_vectors)
        return weight_vectors

    def init_Euler_distance(self):
        weight_vectors = self.weight_vectors
        distance = np.zeros((len(weight_vectors), len(weight_vectors)))
        for i in range(len(weight_vectors)):
            for j in range(len(weight_vectors)):
                distance[i][j] = ((weight_vectors[i] - weight_vectors[j]) ** 2).sum()
        self.Euler_distance = distance
        return distance

    # 生成个体
    def generate_individual(self, genes):
        individual = Individual()
        individual.genes = genes
        individual.genes_size = len(genes)
        individual.fitness = individual.get_fitness(self.objective_functions)
        return individual

    def get_genes(self, individuals_size, genes_size):
        individuals_genes = np.random.random(size=(individuals_size, genes_size))
        for i in range(len(self.variables_range)):
            variable_range = self.variables_range[i]
            individuals_genes[:, i] = individuals_genes[:, i] * (variable_range[1] - variable_range[0]) + variable_range[0]
        return individuals_genes

    # 构造初始种群
    def generate_initial_population(self):
        population = []

        population_genes = self.get_genes(self.population_size, self.genes_size)
        for genes in population_genes:
            individual = self.generate_individual(genes)
            population.append(individual)

        for i in range(self.population_size):
            individual = population[i]
            individual.weight_vector = self.weight_vectors[i]
            individual.objective_functions = self.objective_functions
            distance = self.Euler_distance[i]
            sort_arg = np.argsort(distance)
            neighbors = []
            for j in range(self.T):
                neighbors.append(population[sort_arg[j]])
            individual.neighbors = neighbors
        self.population = population
        return population

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

    # 初始化参考点
    def init_reference_points(self):
        Z = np.full(shape=self.m, fill_value=float('inf'))
        for individual in self.population:
            for i in range(len(individual.fitness)):
                function_val = individual.fitness[i]
                if function_val < Z[i]:
                    Z[i] = function_val
        self.Z = Z

    # 选择操作
    def selection(self, individual):
        neighbors = individual.neighbors
        idx = np.random.randint(0, len(neighbors), size=2)
        k, l = idx[0], idx[1]
        while k == l:
            idx = np.random.randint(0, len(neighbors), size=2)
            k, l = idx[0], idx[1]
        individual_k, individual_l = individual.neighbors[k], individual.neighbors[l]
        return individual_k, individual_l

    # 交叉操作
    def crossover(self, parent1, parent2):
        child1_genes = parent1.genes.copy()
        child2_genes = parent2.genes.copy()
        # print("交叉前child1_genes:", child1_genes)
        # print("交叉前child2_genes:", child2_genes)
        crossover_idx = np.random.randint(0, self.genes_size)
        # print("crossover_idx:", crossover_idx)
        temp_genes = child1_genes[crossover_idx:].copy()
        child1_genes[crossover_idx:] = child2_genes[crossover_idx:]
        child2_genes[crossover_idx:] = temp_genes
        # print("交叉后child1_genes:", child1_genes)
        # print("交叉后child2_genes:", child2_genes)
        child1 = self.generate_individual(child1_genes)
        child2 = self.generate_individual(child2_genes)
        return child1

    # 变异操作
    def mutation(self, individual, variables_range):
        if np.random.rand() < self.mutation_prob:
            genes_size = individual.genes_size
            mutation_idx = np.random.randint(0, genes_size)
            minVal = variables_range[mutation_idx][0]
            maxVal = variables_range[mutation_idx][1]
            individual.genes[mutation_idx] = np.random.rand() * (maxVal - minVal) + minVal
            individual.fitness = individual.get_fitness(self.objective_functions)

    def compare_gte(self, new_individual, old_individual):
        gte_new_individual = np.array(new_individual.fitness - self.Z) * old_individual.weight_vector
        gte_old_individual = np.array(old_individual.fitness - self.Z) * old_individual.weight_vector
        return max(gte_new_individual) - max(gte_old_individual)

    # 更新参考点
    def update_z(self, individual):
        Z = self.Z
        for i in range(len(self.objective_functions)):
            if individual.fitness[i] < Z[i]:
                Z[i] = individual.fitness[i]
        return Z

    def update_EP(self, new_individual):
        if self.EP != []:
            i = 0
            while (i < len(self.EP)):
                if (new_individual.dominate(self.EP[i])):
                    del self.EP[i]
                    i -= 1
                i += 1
            accept_new = True
            for individual in self.EP:
                if (individual.dominate(new_individual) or (individual.fitness == new_individual.fitness).all()):
                    accept_new = False
                    break
            if accept_new:
                self.EP.append(new_individual)
        else:
            self.EP.append(new_individual)

        return self.EP








