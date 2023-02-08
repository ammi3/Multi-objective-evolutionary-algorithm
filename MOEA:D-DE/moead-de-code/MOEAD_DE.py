import numpy as np
import random
from individual import Individual

class MOEAD_DE:
    def __init__(self, H, T, selection_prob, crossover_prob, mutation_prob, objective_functions, variables_range):
        self.H = H
        self.N = None
        self.T = T
        self.genes_size = len(variables_range)
        self.selection_prob = selection_prob
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.objective_functions = objective_functions
        self.variables_range = variables_range

        self.m = len(objective_functions)
        self.population = []
        self.weight_vectors = []
        self.neighbors = []
        self.Z = None

        self.F = 0.5
        self.nr = 2
        self.eta = 20
        self.fronts = []
        self.EP = []

    def generateWeightVectors(self, total, m):
        if m == 1:
            return [[total]]
        weight_vectors = []
        for i in range(1, total - (m - 1) + 1):
            right_vectors = self.generateWeightVectors(total - i, m - 1)
            a = [i]
            for item in right_vectors:
                weight_vectors.append(a + item)
        return weight_vectors

    def getDistance(self, x, y):
        return np.sqrt(np.sum(np.square([x[i] - y[i] for i in range(self.m)])))

    def initializeWeightVectorAndNeighbors(self):
        weight_vectors = self.generateWeightVectors(self.H + self.m, self.m)
        weight_vectors = (np.array(weight_vectors) - 1) / self.H
        self.weight_vectors = weight_vectors
        self.N = len(weight_vectors)

        for i in range(len(weight_vectors)):
            distance = []
            for j in range(len(weight_vectors)):
                if i != j:
                    tup = (j, self.getDistance(self.weight_vectors[i], self.weight_vectors[j]))
                    distance.append(tup)
            distance = sorted(distance, key=lambda x:x[1])
            neighbor = []
            for j in range(self.T):
                neighbor.append(distance[j][0])
            self.neighbors.append(neighbor)

    def initializePopulation(self):
        population = []

        population_genes = np.random.random(size=(self.N, self.genes_size))
        for i in range(self.genes_size):
            variable_range = self.variables_range[i]
            population_genes[:, i] = population_genes[:, i] * (variable_range[1] - variable_range[0]) + \
                                      variable_range[0]

        for gene in population_genes:
            individual = Individual()
            individual.genes = gene
            individual.fitness = individual.get_fitness(self.objective_functions)
            population.append(individual)

        self.population = population

    def fast_non_dominated_sort(self):
        for ind in self.population:
            ind.rank = None
            ind.Sp = []
            ind.np = 0

        F1 = []
        for p in self.population:
            for q in self.population:
                if p.dominate(q):
                    p.Sp.append(q)
                elif q.dominate(p):
                    p.np += 1

            if p.np == 0:
                p.rank = 1
                F1.append(p)

        self.fronts.append(F1)

        i = 0
        while self.fronts[i]:
            Q = []
            for p in self.fronts[i]:
                for q in p.Sp:
                    q.np -= 1
                    if q.np == 0:
                        q.rank = i + 1
                        Q.append(q)

            if Q:
                i += 1
                self.fronts.append(Q)
            else:
                break

        self.EP = self.fronts[0]



    def initializeReferencePoint(self):
        Z = np.full(shape=self.m, fill_value=float('inf'))
        for ind in self.population:
            for i in range(self.m):
                Z[i] = min(Z[i], ind.fitness[i])
        self.Z = Z

    def selection(self, i):
        rand = np.random.random()
        if rand < self.selection_prob:
            P = [j for j in self.neighbors[i]]
        else:
            P = [j for j in range(self.N)]
        return P

    def reproduction(self, P, i):
        random_idx = np.random.randint(0, len(P), size=2)
        k, l = random_idx[0], random_idx[1]
        new_ind = self.crossoverOperation(P, i, k, l)
        self.mutationOperation(new_ind)
        self.repair(new_ind)
        new_ind.fitness = new_ind.get_fitness(self.objective_functions)
        return new_ind


    def crossoverOperation(self, P, r1, r2, r3):
        y = Individual()
        y.genes = np.zeros(shape=self.genes_size)
        for k in range(self.genes_size):
            prob = np.random.random()
            if prob <= self.crossover_prob:
                y.genes[k] = self.population[r1].genes[k] + self.F * (self.population[P[r2]].genes[k] - self.population[P[r3]].genes[k])
            else:
                y.genes[k] = self.population[r1].genes[k]
        return y

    def getLambda(self):
        rand = np.random.random()
        if rand < 0.5:
            lamb = pow(2 * rand, 1 / (self.eta + 1)) - 1
        else:
            lamb = 1 - pow(2 - 2 * rand, 1 / (self.eta + 1))
        return lamb

    def mutationOperation(self, y):
        for k in range(self.genes_size):
            prob = np.random.random()
            if prob <= self.mutation_prob:
                y.genes[k] = y.genes[k] + self.getLambda() * (self.variables_range[k][1] - self.variables_range[k][0])
        return y

    def repair(self, y):
        for i in range(self.genes_size):
            if y.genes[i] > self.variables_range[i][1] or y.genes[i] < self.variables_range[i][0]:
                y.genes[i] = np.random.random() * (self.variables_range[i][1] - self.variables_range[i][0]) + self.variables_range[i][0]

    def updateReferencePoint(self, y):
        for i in range(len(self.objective_functions)):
            if y.fitness[i] < self.Z[i]:
                self.Z[i] = y.fitness[i]

    def updateNeighbors(self, P, i, y):
        c, count = 0, 0
        pre_idx = set()
        while c != self.nr and count != len(P):
            j = np.random.randint(0, len(P))
            if j in pre_idx:
                continue
            pre_idx.add(j)
            if max(np.array(y.fitness - self.Z) * self.weight_vectors[j]) < max(np.array(self.population[P[j]].fitness - self.Z) * self.weight_vectors[j]):
                self.population[P[j]] = y
                c += 1
            count += 1

    def updateEP(self, y):
        if self.EP:
            i = 0
            while i < len(self.EP):
                if y.dominate(self.EP[i]):
                    del self.EP[i]
                    i -= 1
                i += 1
            accept_new = True
            for individual in self.EP:
                if individual.dominate(y) or (individual.fitness == y.fitness).all():
                    accept_new = False
                    break
            if accept_new:
                self.EP.append(y)
        else:
            self.EP.append(y)

        return self.EP










