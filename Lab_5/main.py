import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import scipy.stats as stats
from matplotlib.patches import Ellipse

sizes = [20, 60, 100]
cor = [0, 0.5, 0.9]
r, r_s, r_q = [], [], []
r_2, r_s_2, r_q_2 = [], [], []


def setEllipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2, facecolor=facecolor, **kwargs)
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)
    transf = transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(mean_x, mean_y)
    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def plotEllipse(size):
    mean = [0, 0]
    fig, ax = plt.subplots(1, 3)
    fig.suptitle("n = " + str(size))
    titles = [r'$ \rho = 0$', r'$\rho = 0.5 $', r'$ \rho = 0.9$']
    num = 0
    for j in cor:
        cov = [[1.0, j], [j, 1.0]]
        rv = stats.multivariate_normal.rvs(mean, cov, size=size)
        x = rv[:, 0]
        y = rv[:, 1]
        ax[num].scatter(x, y, s=3)
        setEllipse(x, y, ax[num], edgecolor='navy')
        ax[num].scatter(np.mean(x), np.mean(y), c='aqua', s=3)
        ax[num].set_title(titles[num])
        num += 1
    plt.show()


def quadrant(x, y):
    size = len(x)
    med_x = np.median(x)
    med_y = np.median(y)
    x_new = np.empty(size, dtype=float)
    x_new.fill(med_x)
    x_new = x - x_new
    y_new = np.empty(size, dtype=float)
    y_new.fill(med_y)
    y_new = y - y_new
    n = [0, 0, 0, 0]
    for i in range(size):
        if x_new[i] >= 0 and y_new[i] >= 0:
            n[0] += 1
        if x_new[i] < 0 and y_new[i] > 0:
            n[1] += 1
        if x_new[i] < 0 and y_new[i] < 0:
            n[2] += 1
        if x_new[i] > 0 and y_new[i] < 0:
            n[3] += 1
    return ((n[0] + n[2]) - (n[1] + n[3])) / size


for i in sizes:
    for j in cor:
        for rep in range(1000):
            rvs = stats.multivariate_normal.rvs([0, 0], [[1.0, j], [j, 1.0]], size=i)
            x = rvs[:, 0]
            y = rvs[:, 1]
            r.append(stats.pearsonr(x, y)[0])
            r_2.append(r[rep] ** 2)
            r_s.append(stats.spearmanr(x, y)[0])
            r_s_2.append(r_s[rep] ** 2)
            r_q.append(quadrant(x, y))
            r_q_2.append(r_q[rep] ** 2)
        print("Двумерное нормальное распределение, n = " + str(i) + ", p = " + str(j))
        print(np.around(np.mean(r), decimals=3), np.around(np.mean(r_s), decimals=3), np.around(np.mean(r_q), decimals=3))
        print(np.around(np.mean(r_2), decimals=3), np.around(np.mean(r_s_2), decimals=3),
              np.around(np.mean(r_q_2), decimals=3))
        print(np.around(np.std(r) ** 2, decimals=3), np.around(np.std(r_s) ** 2, decimals=3),
              np.around(np.std(r_q) ** 2, decimals=3), end="\n\n")
        r.clear(), r_s.clear(), r_q.clear(), r_2.clear(), r_s_2.clear(), r_q_2.clear()
    plotEllipse(i)

print("Смесь нормальных распределений")
for i in sizes:
    for rep in range(1000):
        rvs = 0.9 * stats.multivariate_normal.rvs([0, 0], [[1.0, 0.9], [0.9, 1.0]], size=i) \
              + 0.1 * stats.multivariate_normal.rvs([0, 0], [[10, -0.9], [-0.9, 10]], size=i)
        x = rvs[:, 0]
        y = rvs[:, 1]
        r.append(stats.pearsonr(x, y)[0])
        r_2.append(r[rep] ** 2)
        r_s.append(stats.spearmanr(x, y)[0])
        r_s_2.append(r_s[rep] ** 2)
        r_q.append(quadrant(x, y))
        r_q_2.append(r_q[rep] ** 2)
    print("n = " + str(i))
    print(np.around(np.mean(r), decimals=3), np.around(np.mean(r_s), decimals=3), np.around(np.mean(r_q), decimals=3))
    print(np.around(np.mean(r_2), decimals=3), np.around(np.mean(r_s_2), decimals=3),
          np.around(np.mean(r_q_2), decimals=3))
    print(np.around(np.std(r) ** 2, decimals=3), np.around(np.std(r_s) ** 2, decimals=3),
          np.around(np.std(r_q) ** 2, decimals=3))
    r.clear(), r_s.clear(), r_q.clear(), r_2.clear(), r_s_2.clear(), r_q_2.clear()
