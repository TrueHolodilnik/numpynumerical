import numpy as np
import math
import time
import matplotlib.pyplot as plt
import matplotlib

index = 175607
a1 = 5 + 6
a2 = -1
a3 = -1
#N = 9 * 0 * 7
N = 9 * 7

b = np.zeros(shape=(N, 1))

for i in range(1, N):
    b[i] = np.sin(i * (5 + 1))
b[0] = np.sin((N + 1) * (5 + 1))

residuum = 10 ** (-9)
x = np.zeros(shape=(N, 1))

#
#   Matrix functions
#

def generate_A(N, a1, a2, a3):
    A = np.zeros(shape=(N, N))
    for i in range(0, N):
        A[i][i] = a1
        if i < N - 1:
            A[i][i + 1] = a2
            A[i + 1][i] = a2
        if i < N - 2:
            A[i][i + 2] = a3
            A[i + 2][i] = a3
    return A

def calc_norm_res(mat, N):
    norm = 0.0
    for i in range(1, N): norm += math.pow(mat[i], 2)
    norm = math.sqrt(norm)
    return norm

#
#   Jakobi method
#
def Jakobi(N, A, b):
    iterations_jak = 0
    res = np.ones(shape=(N, 1))
    curr_norm_res = 999.0

    L = np.tril(A, -1)
    U = np.triu(A, 1)
    D = np.diag(np.diag(A))

    invD = np.linalg.inv(D)
    tmp1 = np.matmul(invD, b)
    tmp2 = np.matmul(invD, (L + U))

    # Metoda Jakobiego
    start = time.time()
    while (curr_norm_res > residuum):

        res = tmp1 - np.matmul(tmp2, res)
        curr_norm_res = calc_norm_res(np.matmul(A, res) - b, N)
        iterations_jak += 1
        if iterations_jak > 500:
            print("Metoda Jakobiego nie zbiega")
            return

    end = time.time()
    time_jak = end - start
    print("Czas Jakobi: " + str(time_jak) + "\n Iteracje: " + str(iterations_jak))

    return time_jak

#
#   Gauss-Siedl method
#
def GaussSiedl(N, A, b):
    res = np.ones(shape=(N, 1))
    curr_norm_res = 999.0
    iterations_gauss = 0

    L = np.tril(A, -1)
    U = np.triu(A, 1)
    D = np.diag(np.diag(A))

    tmp1 = np.linalg.inv(D + L)
    tmp2 = np.matmul(tmp1, b)

    start = time.time()
    while (curr_norm_res > residuum):

        res = tmp2 - np.matmul(tmp1, np.matmul(U, res))
        curr_norm_res = calc_norm_res(np.matmul(A, res) - b, N)
        iterations_gauss += 1
        if iterations_gauss > 500:
            print("Metoda Gaussa-Siedla nie zbiega")
            return

    end = time.time()
    time_gauss = end - start
    print("Czas Gauss-Siedl: " + str(time_gauss) + "\n Iteracje: " + str(iterations_gauss))

    return time_gauss

A = generate_A(N, a1, a2, a3)

Jakobi(N, A, b)
GaussSiedl(N, A, b)

a1 = 3
a2 = -1
a3 = -1

A = generate_A(N, a1, a2, a3)

Jakobi(N, A, b)
GaussSiedl(N, A, b)

#
#   LU factorization method
#
def LUFactorization(A, b):

    n = A.shape[0]
    U = A.copy()
    L = np.eye(n, dtype=np.double)

    start = time.time()

    for i in range(n):
        factor = U[i + 1:, i] / U[i, i]
        L[i + 1:, i] = factor
        U[i + 1:] -= factor[:, np.newaxis] * U[i]

    #Forward substitution
    n = L.shape[0]
    y = np.zeros_like(b, dtype=np.double);
    y[0] = b[0] / L[0, 0]

    for i in range(1, n):
        y[i] = (b[i] - np.dot(L[i, :i], y[:i])) / L[i, i]

    #Backward substitution
    n = U.shape[0]
    x = np.zeros_like(y, dtype=np.double);
    x[-1] = y[-1] / U[-1, -1]

    for i in range(n - 2, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i:], x[i:])) / U[i, i]

    end = time.time()
    time_LU = end - start
    print("Czas LU: " + str(time_LU))

    res = calc_norm_res(np.matmul(A, x) - b, n)
    print("Residuum LU: " + str(res))

    return time_LU

LUFactorization(A, b)

a1 = 5 + 6
a2 = -1
a3 = -1

N = [100, 500, 1000, 2000, 3000]
time_LU = [0]*5
time_Gauss = [0]*5
time_Jakobi = [0]*5

for i in range(0, 5):

    b = np.zeros(shape=(N[i], 1))
    for j in range(0, N[i] - 1):
        b[j] = np.sin(j * (5 + 1))
    b[0] = np.sin((N[i] + 1) * (5 + 1))
    A = generate_A(N[i], a1, a2, a3)
    x = np.zeros(shape=(N[i], 1))

    time_LU[i] = LUFactorization(A, b)
    time_Gauss[i] = GaussSiedl(N[i], A, b)
    time_Jakobi[i] = Jakobi(N[i], A, b)

plt.plot(time_Gauss, label="Metoda Gaussa-Siedla")
plt.plot(time_Jakobi, label="Metoda Jakobiego")
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(18.5, 10.5)
plt.xlabel('Liczba niewiadomych N')
plt.ylabel('Czas, s')
plt.grid(True)
plt.legend()
plt.show()


