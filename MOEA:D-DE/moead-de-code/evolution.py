from MOEAD_DE import MOEAD_DE
import matplotlib.pyplot as plt
from problems.zdt3 import get_zdt3_params
from problems.zdt4 import get_zdt4_params

objective_functions, variables_range = get_zdt4_params()
genes_size = len(variables_range)
selection_prob = 0.9
crossover_prob = 1
mutation_prob = float(format(1 / genes_size, '.3f'))

iterations = 1000
H = 299
m = 2
T = 20

moead_de = MOEAD_DE(H, T, selection_prob, crossover_prob, mutation_prob, objective_functions, variables_range)

moead_de.initializeWeightVectorAndNeighbors()
population = moead_de.initializePopulation()
moead_de.initializeReferencePoint()
moead_de.fast_non_dominated_sort()

def evolve():
    for iter in range(iterations):
        print('第', iter + 1, '次迭代时，EP大小为：', len(moead_de.EP))
        for i in range(moead_de.N):
            P = moead_de.selection(i)
            new_ind = moead_de.reproduction(P, i)
            moead_de.updateReferencePoint(new_ind)
            moead_de.updateNeighbors(P, i, new_ind)
            moead_de.updateEP(new_ind)


evolve()

function1 = []
function2 = []
for i in moead_de.EP:
    individual_fitness = i.fitness
    function1.append(individual_fitness[0])
    function2.append(individual_fitness[1])

plt.scatter(function1, function2)
# plt.savefig('./pictures/zdt3.png')
plt.show()