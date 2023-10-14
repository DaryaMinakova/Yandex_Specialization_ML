import numpy as np
import pandas as pd


def euclidean_distance(first, second):
    return np.sqrt(np.sum(np.power(first - second, 2)))


def k_nearest_neighbors(n, k):
    ans = []
    for i in range(len(new_table_penguins)):
        if i == n:
            continue
        distance = euclidean_distance(new_table_penguins[i], new_table_penguins[n])
        if len(ans) < k:
            ans.append((distance, new_table_penguins[i]))
            ans = sorted(ans, key=lambda t: t[0])
        elif distance < ans[-1][0]:
            ans[-1] = (distance, new_table_penguins[i])
            ans = sorted(ans, key=lambda t: t[0])
    return ans


penguins_data = pd.read_csv('penguins.csv')
penguins_data.dropna(inplace=True)

new_table_penguins = penguins_data[["bill_length_mm", "bill_depth_mm"]].copy()
new_table_penguins = new_table_penguins.to_numpy()

n, k = int(input()), int(input())
neighbors = k_nearest_neighbors(n, k)
for i in neighbors:
    print(i[1])
