import numpy as np
from nsga2 import *
import matplotlib.pyplot as plt
from problem.kur import get_kur_params
from problem.sch import get_sch_params
from problem.zdt2 import get_zdt2_params
from problem.zdt3 import get_zdt3_params
from problem.zdt4 import get_zdt4_params

# 参数获取
objective_functions, variables_range = get_zdt3_params()
population_size = 200
genes_size = len(variables_range)
crossover_prob = 1
mutation_prob = float(format(1 / genes_size, '.3f'))
# mutation_prob = 0.1
iteration = 250

nsga2 = NSGA2(population_size, genes_size, crossover_prob = crossover_prob, mutation_prob = mutation_prob, objective_functions = objective_functions, variables_range = variables_range)

def evolve():
    population = nsga2.generate_initial_population()
    nsga2.calculate_objectives(population)
    fronts = nsga2.fast_nondominated_sort(population)
    nsga2.calculate_crowding_distance(fronts)
    childen = nsga2.generate_children(population)
    for i in range(iteration):
        population.extend(childen)
        fronts = nsga2.fast_nondominated_sort(population)
        nsga2.calculate_crowding_distance(fronts)
        new_population = []
        front_idx = 0
        while (len(new_population) + len(fronts[front_idx]) <= population_size):
            new_population.extend(fronts[front_idx])
            front_idx += 1
        fronts[front_idx].sort(key=lambda individual: individual.crowding_distance, reverse=True)
        new_population.extend(fronts[front_idx][0: population_size - len(new_population)])
        population = new_population
        fronts = nsga2.fast_nondominated_sort(population)
        nsga2.calculate_crowding_distance(fronts)
        childen = nsga2.generate_children(population)
        print("第", i, "次迭代结束")
    return population, fronts[0]


P, optimal_front = evolve()
print(len(optimal_front))
function1 = []
function2 = []
for i in P:
    individual_fitness = i.fitness
    function1.append(individual_fitness[0])
    function2.append(individual_fitness[1])

plt.scatter(function1, function2)
# plt.savefig('./pictures/zdt4.png')
plt.show()
