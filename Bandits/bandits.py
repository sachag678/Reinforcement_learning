import numpy as np

num_bandits = 10
rewards = [np.random.randint(1, 9) for i in range(num_bandits)]
probs = [np.random.random() for i in range(num_bandits)]
bandits = list(zip(rewards, probs))
qval = [0 for i in range(num_bandits)]
qcounts = [1 for i in range(num_bandits)]

def pull(arm):
    return np.random.binomial(1, p=probs[arm]) 

def greedy(t):
    max_indices = []
    max_val = qval[0]
    max_indices.append(0)
    for i in range(1, len(qval)):
        if(max_val==qval[i]):
            max_indices.append(i)

        if(max_val<qval[i]):
            max_val = qval[i]
            max_indices.clear()
            max_indices.append(i)

    return np.random.choice(max_indices)

def epsilon_greedy(t):
    if(np.random.random()<0.3):
        return np.random.choice([i for i in range(len(qval))])
    else:
        return greedy(t)

def get_ucb(t):
    return np.sqrt(2 * np.log10(t) / (qcounts))


def ucb(t):
    return np.argmax(qval + get_ucb(t))


def run(choose):
    rewards = []
    for i in range(1, 1000):
        choice = choose(i)
        reward = pull(choice)
        rewards.append(reward)
        qval[choice] = qval[choice] + ((reward-qval[choice])/(qcounts[choice]))
        qcounts[choice] += 1

    print(qval)
    print(qcounts)
    print('total reward: {}'.format(np.array(rewards).sum()))
    #plt.plot(np.array(rewards).cumsum())
    #plt.show()

print(bandits)

#run(greedy)
run(epsilon_greedy)
#run(ucb)


