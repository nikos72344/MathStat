import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sts


"""Для двух выборок размерами 20 и 100 элементов, сгенерированных
согласно нормальному закону 𝑁(𝑥, 0, 1), для параметров положения и
масштаба построить асимптотически нормальные интервальные оценки на основе точечных оценок метода максимального 
правдоподобия и классические интервальные оценки на основе статистик 𝜒^2 и Стьюдента. 
В качестве параметра надёжности взять 𝛾 = 0.95"""


size = [20, 100]
alpha = 0.05
interval = []

def MLE(distr):
    m = np.mean(distr)
    s = np.std(distr)
    return m, s


def normal(m, s, distr):
    n = len(distr)
    interval.append(m - s * sts.t.ppf(1 - alpha / 2, n - 1) / (n - 1) ** 0.5)
    interval.append(m + s * sts.t.ppf(1 - alpha / 2, n - 1) / (n - 1) ** 0.5)
    interval.append(s * (n / sts.chi2.ppf((1 - alpha / 2), n - 1)) ** 0.5)
    interval.append(s * (n / sts.chi2.ppf((alpha / 2), n - 1)) ** 0.5)
    return interval


def random(m, s, distr):
    n = len(distr)
    interval.append(m - s * sts.norm.ppf(1 - alpha / 2) / (n ** 0.5))
    interval.append(m + s * sts.norm.ppf(1 - alpha / 2) / (n ** 0.5))
    e = sts.moment(distr, 4) / s ** 4 - 3
    interval.append(s * (1 + 0.5 * sts.norm.ppf(1 - alpha / 2) * (((e + 2) / n) ** 0.5)) ** (-0.5))
    interval.append(s * (1 - 0.5 * sts.norm.ppf(1 - alpha / 2) * (((e + 2) / n) ** 0.5)) ** (-0.5))
    return interval


for i in size:
    distr = np.random.normal(0, 1, i)
    m, s = MLE(distr)
    normal(m, s, distr)
    print("mu: ", interval[0], interval[1], " sigma: ", interval[2], interval[3])
    interval.clear()
    random(m, s, distr)
    print("mu: ", interval[0], interval[1], " sigma: ", interval[2], interval[3])
    interval.clear()
