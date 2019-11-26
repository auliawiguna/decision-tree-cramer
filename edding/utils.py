import numpy as np


def calculate_gini(y) -> float:
    total = len(y)
    f_square = [(np.count_nonzero(y == c) / total) ** 2 for c in np.unique(y)]
    gini = 1 - sum(f_square)

    return gini


def cnt_value(y, n_class):
    unique_y = np.unique(y, return_counts=True)
    value = [0] * n_class
    unique_y = zip(unique_y[0], unique_y[1])

    for cls, cnt in unique_y:
        value[int(cls)] = cnt

    return value


def min_records_cnt(y):
    unique_y = np.unique(y, return_counts=True)
    return min(unique_y[1])


def split_data(X, y, idx, val):
    data = np.append(X, y[:, None], axis=1)
    data_l = data[data[:, idx] <= val]
    data_r = data[data[:, idx] > val]

    return data_l[:, :-1], data_l[:, -1], data_r[:, :-1], data_r[:, -1]


# Pass in g (gini) to avoid repeating calculation
def find_best_split(X, y, g):
    best = 0.0, -1, 0.0, 0.0, 0.0  # gini_gain, index, value, g_l, g_r

    # Iterate over features
    for idx in range(X.shape[1]):
        # Iterate over possible values
        for val in X[:, idx]:
            X_l, y_l, X_r, y_r = split_data(X, y, idx, val)
            g_l = calculate_gini(y_l)
            g_r = calculate_gini(y_r)

            g_gain = g - (g_l + g_r)
            if g_gain > best[0]:
                best = g_gain, idx, val, g_l, g_r

    # Didn't find any split that can reduce gini impurity
    if best[1] < 0:
        return None

    return best[1:]
