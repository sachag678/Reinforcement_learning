import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

count = 1
n = 3
probs = [0.3, 0.7, 0.8]
trials = np.zeros(n)
wins = np.zeros(n)
for i in range(1000):
    p = np.random.beta(1+trials, 1+trials-wins)
    b = np.argmax(p)
    if np.random.rand() <  probs[b]:
        wins[b] += 1
    trials[b] += 1

    if(i%100==0):
        x = np.arange(0, 1, 0.001)
        y0 = beta.pdf(x,1+trials[0], 1+trials[0]-wins[0])
        y1 = beta.pdf(x,1+trials[1], 1+trials[1]-wins[1])
        y2 = beta.pdf(x,1+trials[2], 1+trials[2]-wins[2])
        plt.subplot(5, 2, count, title='Number of trials: {}'.format(i))
        plt.plot(x,y0,"r--",x,y1,"g--",x,y2,"k--")
        count+=1

plt.show()
