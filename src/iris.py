import pandas as pd
import numpy as np
from BareBonesPSO import BareBonesPSO


def fitness(x):
    fit = 0
    for cluster in range(0, k):
        for i in range(0, len(cluster_data[cluster])):
            fit += np.linalg.norm(x[cluster:cluster + d] - cluster_data[cluster].values[i])
    return fit

k = 3               # number of clusters
d = 4               # dimension for each data row
N = k*d             # dimension of each particle
nparticles = 3*N    # number of particles
maxiter = 10*N      # number of iteractions

df = pd.read_csv('/home/mvccouto/Documentos/Mestrado/comp_natural/pso/iris_dataset/iris.data', header=None)
df[4] = df[4].str.strip()

iris_types = df[4].unique()
cluster_data = []
for type in iris_types:
    cluster_data.append(df.loc[df[4] == type].ix[:, 0:3])

bounds = []
for i in range(0, k):
    for j in range(0, d):
        max_value = cluster_data[i][j].max()
        min_value = cluster_data[i][j].min()
        bounds.append([min_value, max_value])

print(bounds)
bbPSO = BareBonesPSO(bounds, N, nparticles)
print(bbPSO.optimize(maxiter, fitness))

