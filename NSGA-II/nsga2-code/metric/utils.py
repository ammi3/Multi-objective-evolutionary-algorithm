import numpy as np

def get_distance(individual1, individual2):
    distance = 0
    for i in range(len(individual1.fitness)):
        distance += np.sqrt((individual2.fitness[i] - individual1.fitness[i]) ** 2)
    return distance
