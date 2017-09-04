import pandas as pd
import numpy as np
from BareBonesPSO import BareBonesPSO


nparticles = 50

def main():

    df = pd.read_csv('/home/mvccouto/Documentos/Mestrado/comp_natural/pso/iris_dataset/iris.data', header=None)
    df[4] = df[4].str.strip()

    iris_types = df[4].unique()
    for type in iris_types:
        cluster_data = df.loc[df[4] == type].ix[:, 0:3]
        max_value = cluster_data.ix[:, 0:3].values.max()
        min_value = cluster_data.ix[:, 0:3].values.min()

        def func(x):
            v = 0
            for row in cluster_data.values:
                v += np.linalg.norm(x-row)
            return v

        bbPSO = BareBonesPSO([min_value, max_value], 4, nparticles)
        print(bbPSO.optimize(30, func))

if __name__ == "__main__":
    main()