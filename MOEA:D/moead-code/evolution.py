from moead import MOEAD
import matplotlib.pyplot as plt
from problem.zdt3 import get_zdt3_params
from problem.zdt4 import get_zdt4_params

# 参数获取
objective_functions, variables_range = get_zdt3_params()
genes_size = len(variables_range)
crossover_prob = 1
mutation_prob = float(format(1 / genes_size, '.3f'))
# mutation_prob = 0.3
iterations = 1000

H = 249
m = 2
T = 20

moead = MOEAD(H, m, T, genes_size, mutation_prob, objective_functions, variables_range)

moead.init_weight_vectors()
moead.init_Euler_distance()
population = moead.generate_initial_population()
moead.init_reference_points()
fronts = moead.fast_nondominated_sort(population)
moead.EP = fronts[0]

def evolve():
    for i in range(iterations):
        print('第', i + 1, '次迭代时，EP大小为：', len(moead.EP))
        for individual in population:
            parent1, parent2 = moead.selection(individual)
            child1 = moead.crossover(parent1, parent2)
            moead.mutation(child1, variables_range)
            moead.update_z(child1)
            for neighbor in individual.neighbors:
                if moead.compare_gte(child1, neighbor) <= 0:
                    neighbor.genes = child1.genes
                    neighbor.fitness = child1.fitness
            moead.update_EP(child1)

evolve()

function1 = []
function2 = []
for i in moead.EP:
    individual_fitness = i.fitness
    function1.append(individual_fitness[0])
    function2.append(individual_fitness[1])

plt.scatter(function1, function2)
plt.savefig('./pictures/zdt3.png')
plt.show()



