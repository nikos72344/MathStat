import numpy
import scipy.stats as stats
from numpy.random import normal, standard_cauchy, laplace, poisson, uniform
import numpy as np
from tabulate import tabulate

rows = []
header = ["name", "mean", "median", "z_R", "z_Q", "z_tr"]
mean, med_x, z_R, z_Q, z_tr = [], [], [], [], []

def fillTable(distr):
    mean.append(np.mean(distr))
    med_x.append(np.median(distr))
    sorted = np.sort(distr)
    z_R.append((sorted[0] + sorted[-1])/2)
    z_Q.append((np.percentile(distr, 1 / 4 * 100) + np.percentile(distr, 3 / 4 * 100)) / 2)
    z_tr.append(stats.trim_mean(distr, 0.25))

def addRow(distrName, n):
    rows.append([distrName + " E(z) " + "n = " + str(n),
                 np.around(np.mean(mean), decimals=6),
                 np.around(np.mean(med_x), decimals=6),
                 np.around(np.mean(z_R), decimals=6),
                 np.around(np.mean(z_Q), decimals=6),
                 np.around(np.mean(z_tr), decimals=6)])
    rows.append([distrName + " D(z) " + "n = " + str(n),
                 np.around(np.std(mean) * np.std(mean), decimals=6),
                 np.around(np.std(med_x) * np.std(med_x), decimals=6),
                 np.around(np.std(z_R) * np.std(z_R), decimals=6),
                 np.around(np.std(z_Q) * np.std(z_Q), decimals=6),
                 np.around(np.std(z_tr) * np.std(z_tr), decimals=6)])
    mean.clear()
    med_x.clear()
    z_R.clear()
    z_Q.clear()
    z_tr.clear()

def varRange(n):
    for i in range(1000):
        fillTable(normal(0, 1, n))
    addRow("normal", n)
    for i in range(1000):
        fillTable(standard_cauchy(n))
    addRow("cauchy", n)
    for i in range(1000):
        fillTable(laplace(0, 2**(-0.5), n))
    addRow("laplace", n)
    for i in range(1000):
        fillTable(poisson(10, n))
    addRow("poisson", n)
    for i in range(1000):
        fillTable(uniform(-1*(3**0.5),3**0.5, n))
    addRow("uniform", n)

size = [10, 100, 1000]
for i in range(len(size)):
    varRange(size[i])
    print(tabulate(rows, header, tablefmt="latex"))
    rows.clear()
