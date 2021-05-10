import numpy as np
import matplotlib.pyplot as plt


def mnk(x, y):
    betta1 = (np.mean(x * y) - np.mean(x) * np.mean(y)) / (np.mean(x * x) - np.mean(x) ** 2)
    betta0 = np.mean(y) - np.mean(x) * betta1
    return betta0, betta1


def mnm(x, y):
    r_Q = np.mean(np.sign(x - np.median(x)) * np.sign(y - np.median(y)))
    betta1_R = r_Q * (np.percentile(y, 75) - np.percentile(y, 25)) / (np.percentile(x, 75) - np.percentile(x, 25))
    betta0_R = np.median(y) - betta1_R * np.median(x)
    return betta0_R, betta1_R


def distance(y1, y2):
    sum = 0
    for i in range(len(y1)):
        sum += ((y1[i] - y2[i]) * (y1[i] - y2[i]))
    return sum

def plot(x, y):
    plt.scatter(x, y, marker='o', c='green')
    plt.plot(x, 2 + 2 * x, color='gold', label='Эталон')

    coef = mnk(x, y)
    dist = distance(coef[1] * x + coef[0], 2 + 2 * x)
    print("МНК" + "\nbetta0: " + str(coef[0]) + ", betta1: " + str(coef[1]))
    print("distance: " + str(dist))
    plt.plot(x, coef[1] * x + coef[0], color='maroon',  label='МНК')

    coef = mnm(x, y)
    dist = distance(coef[1] * x + coef[0], 2 + 2 * x)
    print("МНМ" + "\nbetta0_R: " + str(coef[0]) + ", betta1_R: " + str(coef[1]))
    print("distance: " + str(dist))
    plt.plot(x, coef[1] * x + coef[0], color='midnightblue', label='МНМ')
    plt.legend()
    plt.show()
    print("\n")


eps = np.random.normal(0, 1, size=20)
x = np.linspace(-1.8, 2, 20)
y = 2 + 2 * x + eps
print("Без возмущений")
plot(x, y)

print("С возмущениями")
y[0] += 10
y[-1] -= 10
plot(x, y)
