import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from statsmodels.distributions.empirical_distribution import ECDF
import seaborn as sns

size = [20, 60, 100]
arrays = []
a, b, step = (0, 0, 0.01)

distribution = {
     'Normal': lambda n: np.random.normal(0, 1, n),
     'Cauchy': lambda n: np.random.standard_cauchy(n),
     'Laplace': lambda n: np.random.laplace(0, 2 ** 0.5 / 2, n),
     'Poisson': lambda n: np.random.poisson(10, n),
     'Uniform': lambda n: np.random.uniform(-3 ** 0.5, 3 ** 0.5, n)
}


def getDistribution(name, n):
    return distribution.get(name)(n)


cdf = {
     'Normal': lambda arr: stats.norm.cdf(arr),
     'Cauchy': lambda arr: stats.cauchy.cdf(arr),
     'Laplace': lambda arr: stats.laplace.cdf(arr),
     'Poisson': lambda arr: stats.poisson.cdf(arr, 10),
     'Uniform': lambda arr: stats.uniform.cdf(arr)
}


def getCdf(name, arr):
    return cdf.get(name)(arr)


pdf = {
     'Normal': lambda arr: stats.norm.pdf(arr, 0, 1),
     'Cauchy': lambda arr: stats.cauchy.pdf(arr),
     'Laplace': lambda arr: stats.laplace.pdf(arr, 0, 1 / 2 ** 0.5),
     'Poisson': lambda arr: stats.poisson.pmf(10, arr),
     'Uniform': lambda arr: stats.uniform.pdf(arr, -3 ** 0.5, 2 * 3 ** 0.5)
}


def getPdf(name, arr):
    return pdf.get(name)(arr)


for name in distribution.keys():
    for i in size:
        arrays.append(getDistribution(name, i))

    if name == 'Poisson':
        a, b, step = (6, 14, 0.1)
    else:
        a, b, step = (-4, 4, 0.01)
    array_global = np.arange(a, b, step)

    for j in range(len(arrays)):
        for elem in arrays[j]:
            if elem < a or elem > b:
                arrays[j] = np.delete(arrays[j], list(arrays[j]).index(elem))
        plt.subplot(1, 3, j+1)
        plt.title(name + ' n = ' + str(size[j]))
        plt.plot(array_global, getCdf(name, array_global), color='blue', linewidth=0.8)
        array_ex = np.linspace(a, b)
        ecdf = ECDF(arrays[j])
        y = ecdf(array_ex)
        plt.step(array_ex, y, color='black')
        plt.xlabel('x')
        plt.ylabel('F(x)')
    plt.show()

    k = 1
    for array in arrays:
        titles = [r'$h = \frac{h_n}{2}$', r'$h = h_n$', r'$h = 2 * h_n$']
        fig, ax = plt.subplots(1, 3)
        plt.subplots_adjust(wspace=0.5)
        l = 0
        for bandwidth in [0.5, 1, 2]:
            kde = stats.gaussian_kde(array, bw_method='silverman')
            h_n = kde.factor
            fig.suptitle(name + ' n = ' + str(size[k-1]))
            ax[l].plot(array_global, getPdf(name, array_global), color='blue', alpha=0.5, label='density')
            ax[l].set_title(titles[l])
            sns.kdeplot(array, ax=ax[l], bw=h_n * bandwidth, label='kde')
            ax[l].set_xlabel('x')
            ax[l].set_ylabel('f(x)')
            ax[l].set_ylim([0, 1])
            ax[l].set_xlim([a, b])
            ax[l].legend()
            l += 1
        plt.show()
        k += 1

    arrays.clear()
