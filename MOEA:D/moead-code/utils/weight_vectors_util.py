import numpy as np

def generate_weight_vectors(total, m):
    if m == 1:
        return [[total]]
    weight_vectors = []
    for i in range(1, total - (m - 1) + 1):
        right_vectors = generate_weight_vectors(total - i, m - 1)
        a = [i]
        for item in right_vectors:
            weight_vectors.append(a + item)
    return weight_vectors

