from numpy import quantile, around
from numpy.random import normal, standard_cauchy, laplace, poisson, uniform
import math
import matplotlib.pyplot as plt
from tabulate import tabulate

distributions = {
     "normal": lambda n: normal(0, 1, n),
     "cauchy": lambda n: standard_cauchy(n),
     "laplace": lambda n: laplace(0, 2**(-0.5), n),
     "pois": lambda n: poisson(10, n),
     "uniform": lambda n: uniform(-math.sqrt(3), math.sqrt(3), n)
}


def getDistribution(distrName, n):
    return distributions.get(distrName)(n)


def theoreticalProb(sample):
    min = quantile(sample, 0.25) - 1.5 * (quantile(sample, 0.75) - quantile(sample, 0.25))
    max = quantile(sample, 0.75) + 1.5 * (quantile(sample, 0.75) - quantile(sample, 0.25))
    return min, max


def ejectionNum(rv, min, max):
    ejection = 0
    for elem in rv:
        if elem < min or elem > max:
            ejection += 1
    return ejection


def lab_3():
    for name in distributions.keys():
        rv20 = getDistribution(name, 20)
        rv100 = getDistribution(name, 100)
        plt.boxplot(x = (rv20, rv100), vert=False, labels=[20, 100])
        plt.xlabel("x")
        plt.ylabel("n")
        plt.title(name)
        plt.show()

    rows = []

    for name in distributions.keys():
        eject20 = 0
        eject100 = 0
        for i in range(1000):
            rv20 = getDistribution(name, 20)
            rv100 = getDistribution(name, 100)
            min20, max20 = theoreticalProb(rv20)
            min100, max100 = theoreticalProb(rv100)
            eject20 += ejectionNum(rv20, min20, max20)
            eject100 += ejectionNum(rv100, min100, max100)

        rows.append([name + ", n = 20", around(eject20 / 1000 / 20, decimals=2)])
        rows.append([name + ", n = 100", around(eject100 / 1000 / 100, decimals=2)])

    print(tabulate(rows))
    print("\n")

lab_3()
