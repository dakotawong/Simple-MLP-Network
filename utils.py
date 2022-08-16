import numpy as np

# Sigmoid Activation Function
def sigmoid(x):
    return 1/(1 + np.exp(-x))

# Softmax Activation Function
def softmax(X):
    exps = np.exp(X)
    return exps / np.sum(exps)

# Squared Error
def squared_error_loss(target, output):
    loss = np.sum((target-output)**2)
    return loss

# Derrivative of sigmoid with respect to input
def sigmoid_gradient(x):
    return sigmoid(x)*(1-sigmoid(x))

# Derrivative of softmax 
def softmax_gradient(output):
    temp = output
    return temp * (1 - temp)