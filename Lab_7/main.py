import numpy as np
import scipy.stats as sts
import math


distribution = {
    "Normal": lambda size, mu, sigma: np.random.normal(mu, sigma, size=size),
    "Laplace": lambda size, mu, sigma: np.random.laplace(mu, sigma, size=size),
    "Uniform": lambda size, mu, sigma: np.random.uniform(mu-(3 ** (1/2))*sigma, mu+(3 ** (1/2))*sigma, size=size)
}


def getDistribution(name, size, mu, sigma):
    return distribution.get(name)(size, mu, sigma)


cdf = {
    "Normal": lambda x, mu, sigma: sts.norm.cdf(x, loc=mu, scale=sigma),
    "Laplace": lambda x, mu, sigma: sts.laplace.cdf(x, loc=mu, scale=sigma / (2 ** (1/2))),
    "Uniform": lambda x, mu, sigma: x/(2*(3 ** (1/2))*sigma)
}


def getCdf(name, x, mu, sigma):
    return cdf.get(name)(x, mu, sigma)


def MLE(size):
    norm = getDistribution("Normal", size, 0, 1)
    mu = np.mean(norm)
    sigma = np.std(norm)
    return mu, sigma


def chi2(name, number):
    mu, sigma = MLE(number)
    print("mu = ", mu, ", sigma = ", sigma, "\n")
    if name == "Normal":
        size = int(np.round(1.72 * (number ** (1 / 3))))

    elif name == "Laplace" or name == "Uniform":
        size = int(np.round(1 + 3.3 * math.log(number, 10)))


    quantile = sts.chi2.isf(0.05, size - 1)
    n = np.zeros(size)
    p = np.zeros(size)
    x = np.linspace(-1.5, 1.5, size - 1)
    print("x_i: ", x, "\n")
    distr = getDistribution(name, number, mu, sigma)

    for elem in distr:
        if elem <= x[0]:
            n[0] += 1
        elif elem > x[size-2]:
            n[size-1] += 1
        for i in range(size-2):
            if ((elem > x[i]) & (elem <= x[i + 1])):
                n[i + 1] += 1

    for i in range(size-2):
        p[i+1] = getCdf(name, x[i+1], mu, sigma) - getCdf(name, x[i], mu, sigma)
    p[0] = getCdf(name, x[0], mu, sigma)
    p[size-1] = 1 - getCdf(name, x[size-2], mu, sigma)
    return x, n, p, quantile


def printCharacters(name, number):
    print(name, "distribution, size = ", number)
    x, n, p, quantile = chi2(name, number)
    print("n_i: ", np.around(n, decimals=3), "\nsum n_i = ", np.around(np.sum(n), decimals=3), "\n")
    print("p_i: ", np.around(p, decimals=3), "\nsum p_i = ",  np.around(np.sum(p), decimals=3), "\n")
    print("n*p_i: ", np.around(number * p, decimals=3), "\nsum n*p_i= ", np.around(np.sum(number * p), decimals=3), "\n")
    print("n_i - n*p: ", np.around(n - number * p, decimals=3), "\nsum n_i - n*p = ", np.around(np.sum(n - number * p), decimals=3), "\n")
    res = np.around(((n - number * p) ** 2) / (number * p), decimals=3)
    chi_v = 0
    for i in res:
        chi_v += i
    print("Result: ", np.around(((n - number * p) ** 2) / (number * p), decimals=3))
    print("Summation: ", chi_v)
    print("Quantile", np.around(quantile, decimals=3))

    if chi_v < quantile:
        print("Гипотеза H_0 принимается\n")
    else:
        print("Гипотеза H_0 отвергается\n")


printCharacters("Normal", 100)
printCharacters("Laplace", 20)
printCharacters("Uniform", 20)
