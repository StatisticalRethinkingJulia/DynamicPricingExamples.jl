import pymc3 as pm
import numpy as np
from tabulate import tabulate
from scipy.optimize import linprog
import scipy.stats as stats 
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.animation as animation

d0 = [20, 28, 24, 20, 23]                # observed demand samples

with pm.Model() as m:
    d = pm.Gamma('theta', 1, 1)          # prior distribution
    pm.Poisson('d0', d, observed = d0)   # likelihood
    samples = pm.sample(10000)           # draw samples from the posterior

    
seaborn.distplot(samples.get_values('theta'), fit=scipy.stats.gamma)

p = np.linspace(10, 16)   # price range
d_means = np.exp(s.log_b + s.a * np.log(p).reshape(-1, 1))

plt.plot(p, d_means, c = 'k', alpha = 0.01)
plt.plot(p0, d0, 'o', c = 'r')
plt.show()

