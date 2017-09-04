import pandas as pd
import numpy as np
from BareBonesPSO import BareBonesPSO


def fitness(x):
    fit = 0
    for i in range(0, len(df.values)):
        dists = []
        item = df.values[i]
        for cluster in range(0, k):
            dists.append(np.linalg.norm(x[cluster:cluster + d] - item))
        fit += min(dists)
    return fit

k = 3               # number of clusters
d = 4               # dimension for each data row
N = k*d             # dimension of each particle
nparticles = 3*N    # number of particles
maxiter = 10*N      # number of iteractions

df = pd.read_csv('./datasets/iris.data', header=None).ix[:, 0:3]

bounds = []
for j in range(0, N):
    max_value = df.values.max()
    min_value = df.values.min()
    bounds.append([min_value, max_value])

bbPSO = BareBonesPSO(bounds, N, nparticles)
print(bbPSO.optimize(maxiter, fitness))

