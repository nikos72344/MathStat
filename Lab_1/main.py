import numpy as np
from scipy.stats import norminvgauss, laplace, poisson, cauchy, uniform
import matplotlib.pyplot as plt
import math

size = [10, 50, 1000]


def norminvgaussFunc():
    a, b = 1, 0

    for i in range(len(size)):
        n = size[i]
        fig, ax = plt.subplots(1, 1)
        ax.set_title("Нормальное распределение, n = " + str(n))
        x = np.linspace(norminvgauss.ppf(0.01, a, b), norminvgauss.ppf(0.99, a, b), 100)
        ax.plot(x, norminvgauss.pdf(x, a, b), 'b-', lw=5, alpha=0.6)
        # rv = norminvgauss(a, b)
        # ax.plot(x, rv.pdf(x), 'k-', lw = 2, label = "frozen pdf")
        r = norminvgauss.rvs(a, b, size=n)
        ax.hist(r, density=True, histtype='stepfilled', alpha=0.2)
        plt.show()


def laplaceFunc():
    for i in range(len(size)):
        n = size[i]
        fig, ax = plt.subplots(1, 1)
        ax.set_title("Распределение Лапласа, n = " + str(n))
        x = np.linspace(laplace.ppf(0.01), laplace.ppf(0.99), 100)
        ax.plot(x, laplace.pdf(x), 'b-', lw=5, alpha=0.6)
        r = laplace.rvs(size=n)
        ax.hist(r, density=True, histtype='stepfilled', alpha=0.2)
        plt.show()


def poissonFunc():
    mu = 10

    for i in range(len(size)):
        n = size[i]
        fig, ax = plt.subplots(1, 1)
        ax.set_title("Распределение Пуассона, n = " + str(n))
        x = np.arange(poisson.ppf(0.01, mu), poisson.ppf(0.99, mu))
        ax.plot(x, poisson(mu).pmf(x), 'b-', ms=8)
        r = poisson.rvs(mu, size=n)
        ax.hist(r, density=True, histtype='stepfilled', alpha=0.2)
        plt.show()


def cauchyFunc():
    for i in range(len(size)):
        n = size[i]
        fig, ax = plt.subplots(1, 1)
        ax.set_title("Распределение Коши, n = " + str(n))
        x = np.linspace(cauchy.ppf(0.01), cauchy.ppf(0.99), 100)
        ax.plot(x, cauchy.pdf(x), 'b-', lw=5, alpha=0.6)
        r = cauchy.rvs(size=n)
        ax.hist(r, density=True, histtype='stepfilled', alpha=0.2)
        plt.show()


def uniformFunc():
    for i in range(len(size)):
        n = size[i]
        fig, ax = plt.subplots(1, 1)
        ax.set_title("Нормальное распределение, n = " + str(n))
        x = np.linspace(uniform(loc=-math.sqrt(3), scale=2 * math.sqrt(3)).ppf(0.01),
                        uniform(loc=-math.sqrt(3), scale=2 * math.sqrt(3)).ppf(0.99), 100)
        ax.plot(x, uniform(loc=-math.sqrt(3), scale=2 * math.sqrt(3)).pdf(x), 'b-', lw=5, alpha=0.6)
        r = uniform.rvs(size=n, loc=-math.sqrt(3), scale=2 * math.sqrt(3))
        ax.hist(r, density=True, histtype='stepfilled', alpha=0.2)
        plt.show()


norminvgaussFunc()
laplaceFunc()
poissonFunc()
cauchyFunc()
uniformFunc()
