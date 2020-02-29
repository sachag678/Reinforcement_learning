import numpy as np

true_weights = np.array([2, -1, 3])
alpha = 1

start_weights = np.random.rand(3,1)

for i in range(100):

    feature = np.random.rand(1,3)
    output = feature.dot(start_weights)
    error = ((output - feature.dot(true_weights))*feature).transpose()

    start_weights = start_weights - alpha*error # line search update

print(start_weights)




