import numpy as np


def gradient_descent(func, start_point, gamma, epsilon, steps):
    ans = [np.array([start_point])]
    delta_x = 1e-9
    if steps == 0:
        while True:
            x = ans[-1][0] - gamma * ((func(ans[-1][0] + delta_x) - func(ans[-1][0])) / delta_x)
            ans.append(np.array([x]))
            if np.abs(func(ans[-1][0]) - func(ans[-2][0])) < epsilon:
                break
    else:
        for i in range(steps):
            x = ans[-1][0] - gamma * ((func(ans[-1][0] + delta_x) - func(ans[-1][0])) / delta_x)
            ans.append(np.array([x]))
    return np.round(np.array(ans), 3)
