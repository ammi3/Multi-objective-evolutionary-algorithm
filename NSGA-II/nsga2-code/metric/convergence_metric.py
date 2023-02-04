from metric.utils import get_distance


def convergence_metric(optimal_front, truePF):
    total_distance = 0
    for individual in optimal_front:
        min_distance = float('inf')
        for other in truePF:
            min_distance = min(min_distance, get_distance(individual, other))
        total_distance += min_distance
    return total_distance / len(optimal_front)
