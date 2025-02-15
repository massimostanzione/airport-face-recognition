from math import log

from scipy.stats import truncnorm, lognorm
import numpy as np
import matplotlib.pyplot as plt

from DES_Python.rvgs import Lognormal, TruncatedNormal

if __name__ == '__main__':
    # Parametri per la Normale Troncata
    mu_norm = .8      # media leggermente spostata
    sigma_norm = .025    # deviazione standard
    lower_norm, upper_norm = .75,1
        # # Normalizzazione per truncnorm
    #a, b = (lower_norm - mu_norm) / sigma_norm, (upper_norm - mu_norm) / sigma_norm
    samples_norm=[]
    for i in range(10000):
        samples_norm.append(TruncatedNormal(.8, .05, .78, 18.0))#truncnorm.rvs(a, b, loc=mu_norm, scale=sigma_norm, size=10000)

    # Parametri per la Log-Normale Troncata
    mu_lognorm = np.log(0.5)  # media logaritmica
    sigma_lognorm = 0.05      # piccola deviazione per non avere una coda troppo pesante
    samples_lognorm=[]
    b = 0.025  # Coda destra leggera
    a = log(0.8) + b ** 2  # a ≈ -0.6831
    for i in range(10000):
        samples_lognorm.append(Lognormal(a,b))
    #samples_lognorm = lognorm.rvs(sigma_lognorm, scale=np.exp(mu_lognorm), size=10000)

    # Troncatura della lognormale tra 475 e 575
    #samples_lognorm = samples_lognorm[(samples_lognorm >= 475) & (samples_lognorm <= 575)]

    # Plot delle due distribuzioni
    plt.figure()
    plt.hist(samples_norm, bins=50, density=True, alpha=0.6, color='blue', label='Normale Troncata')
    plt.hist(samples_lognorm, bins=50, density=True, alpha=0.6, color='green', label='Log-Normale Troncata')
    plt.axvline(.8, color='red', linestyle='dashed', linewidth=2, label='500 ms (Target)')
    plt.title("Confronto tra Normale Troncata e Log-Normale Troncata")
    plt.xlabel("Tempo di Servizio (ms)")
    plt.ylabel("Densità")
    plt.legend()
    plt.show()
