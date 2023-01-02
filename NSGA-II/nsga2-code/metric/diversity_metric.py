from metric.utils import get_distance


def get_df_and_dl(optimal_front, truePF):
    return 0

def diversity_metric(optimal_front, truePF):
    distance = []
    for i in range(len(optimal_front) - 1):
        distance.append(get_distance(optimal_front[i], optimal_front[i + 1]))
    avg_distance = sum(distance) / len(distance)
    df, dl = get_df_and_dl(optimal_front, truePF)
    offset = 0
    for i in range(len(optimal_front) - 1):
        offset += distance[i] - avg_distance
    return (df + dl + offset) / (df + dl + (len(optimal_front) - 1) * avg_distance)
