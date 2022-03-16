
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import random
import time

def Lagrange(data, distance, n):
    r = 0.0
    for i in range(n):
        a = 1.0
        for j in range(n):
            if i != j:
                a *= (distance - data[j][0]) / (data[i][0] - data[j][0])
        r += a * data[i][1]

    return r

def Cubic(data, xd, nodes_number):

    N = 4 * (nodes_number - 1)
    M = np.zeros((N, N))
    b = [0]*N
    x = [1]*N

    M[0][0] = 1
    b[0] = data[0][1]

    h = data[1][0] - data[0][0]
    M[1][0] = 1
    M[1][1] = h
    M[1][2] = pow(h, 2)
    M[1][3] = pow(h, 3)
    b[1] = data[1][1]

    M[2][2] = 1
    b[2] = 0

    h = data[nodes_number - 1][0] - data[nodes_number - 2][0]
    M[3][4 * (nodes_number - 2) + 2] = 2
    M[3][4 * (nodes_number - 2) + 3] = 6 * h
    b[3] = 0

    for i in range(1, nodes_number - 1):
        h = data[i][0] - data[i - 1][0]

        M[4 * i][4 * i] = 1
        b[4 * i] = data[i][1]

        M[4 * i + 1][4 * i] = 1
        M[4 * i + 1][4 * i + 1] = h
        M[4 * i + 1][4 * i + 2] = pow(h, 2)
        M[4 * i + 1][4 * i + 3] = pow(h, 3)
        b[4 * i + 1] = data[i + 1][1]

        M[4 * i + 2][4 * (i - 1) + 1] = 1
        M[4 * i + 2][4 * (i - 1) + 2] = 2 * h
        M[4 * i + 2][4 * (i - 1) + 3] = 3 * pow(h, 2)
        M[4 * i + 2][4 * i + 1] = -1
        b[4 * i + 2] = 0

        M[4 * i + 3][4 * (i - 1) + 2] = 2
        M[4 * i + 3][4 * (i - 1) + 3] = 6 * h
        M[4 * i + 3][4 * i + 2] = -2
        b[4 * i + 3] = 0

    x = np.linalg.solve(M, b)

    r = 0
    for i in range(0, nodes_number - 1):
        r = 0
        if (xd >= data[i][0]) and (xd <= data[i + 1][0]):
            for j in range(0, 4):
                h = (xd - data[i][0])
                r += x[4 * i + j] * pow(h, j)
            break

    return r

def Interpolation(n, nodes):

    results_1 = [0] * (int(nodes[n - 1][0]) + 1)
    results_2 = [0] * (int(nodes[n - 1][0]) + 1)

    i = nodes[0][0] + 10

    time_l = 0
    time_c = 0

    while i < nodes[n - 1][0]:

        start = time.time()
        results_1[int(i - nodes[0][0])] = Lagrange(nodes, i, n)
        end = time.time()
        time_l += end - start
        start = time.time()
        results_2[int(i - nodes[0][0])] = Cubic(nodes, i, n)
        end = time.time()
        time_c += end - start
        i += 300

    print("Lagrange time: " + str(time_l))
    print("Spline time: " + str(time_c))

    res_1_x = [0] * n
    res_1_y = [0] * n
    res_2_x = [0] * n
    res_2_y = [0] * n

    for i in range(len(results_1)):
        if results_1[i] != 0:
            res_1_x.append(i)
            res_1_y.append(results_1[i])
    for i in range(len(results_2)):
        if results_2[i] != 0:
            res_2_x.append(i)
            res_2_y.append(results_2[i])

    return res_1_x, res_1_y, res_2_x, res_2_y


track_1 = pd.read_csv('canion.csv')
track_2 = pd.read_csv('everest.csv')

data = track_2.values
length = len(track_2.values)

coeffs = [80, 60, 50, 40]

for j in range(0, 4):

    for s in range(0, 2):

        nodes_number = int(length/coeffs[j])
        nodes = [[0]*2 for z in range(nodes_number)]
        steps = [0]*1000

        if s == 1:
            for l in range(0, 1000):
                steps[l] = coeffs[j]
        else:
            for l in range(0, 1000):
                steps[l] = random.randint(int(coeffs[j]*0.6), int(coeffs[j]*1.2))

        k = 0
        for l in range(0, nodes_number):
            nodes[l][0] = data[k][0]
            nodes[l][1] = data[k][1]
            k += steps[l]

        res_1_x, res_1_y, res_2_x, res_2_y = Interpolation(nodes_number, nodes)

        plt.plot(res_1_x, res_1_y, label="Rezultaty aproksymowane (Lagrange)", zorder = 0)
        plt.plot(res_2_x, res_2_y, label="Rezultaty aproksymowane (Cubiq splines)", zorder = 0)
        d_x = [0] * length
        d_y = [0] * length
        for z in range(0, length):
            d_x[z] = data[z][0]
            d_y[z] = data[z][1]
        plt.plot(d_x, d_y, label="Oryginalna trasa", zorder = 5)
        n_x = [0]*nodes_number
        n_y = [0]*nodes_number
        for z in range(0, nodes_number):
            n_x[z] = nodes[z][0]
            n_y[z] = nodes[z][1]
        if s == 1: plt.scatter(n_x, n_y, label="Punkty do aproksymacji: " + str(nodes_number) + " punktów. (Linear)", c = "red", zorder = 10)
        else: plt.scatter(n_x, n_y, label="Punkty do aproksymacji: " + str(nodes_number) + " punktów. (Random)", c="red", zorder=10)
        fig = matplotlib.pyplot.gcf()
        fig.set_size_inches(9.5, 5.5)
        plt.grid(True)
        plt.legend()
        plt.show()

