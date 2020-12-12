import matplotlib.pyplot as plt
import numpy as np

def quantiles_plot(y_data, y_sampled, areas, label, lw):
    T, N = y_sampled.shape
    # y_sampled is TimexNsamples

    y_sorted = np.sort(y_sampled, axis = 1)

    colors = ['g', 'tab:orange', 'r', 'c', 'm', 'y_sampled', 'k']
    i = 0
    for a in areas:
        c = colors[i]
        alpha1 = (1 - a)/2
        alpha2 = 1- alpha1

        y_up = y_sorted[:,[int(N*alpha1)]]
        y_down = y_sorted[:,int(N*alpha2)]
        plt.plot(y_up, color = c, alpha = 0.7)
        plt.plot(y_down, color = c, alpha = 0.7, label = "{} confidence interval".format(a))
        i = i+1

    plt.plot(y_data, color='black', label="data", linewidth=lw)
    plt.legend()