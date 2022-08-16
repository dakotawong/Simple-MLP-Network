import numpy as np

from MLP import *

# Load Data
train_inputs = np.genfromtxt('data/train_data.csv', delimiter=',')
train_labels = np.genfromtxt('data/train_labels.csv', delimiter=',')

print(train_inputs.shape)
print(train_labels.shape)

# Shuffle Data
idx = np.random.permutation(len(train_labels))
X, Y = train_inputs[idx], train_labels[idx]
X = np.expand_dims(X, axis=-1)
Y = np.expand_dims(Y, axis=-1)

# Split Data
test_size = 0.2
test_len = int(len(train_labels) * test_size)
X_test, X_train = X[:test_len], X[test_len:]
Y_test, Y_train = Y[:test_len], Y[test_len:] 

# Training the Model
model = MLP(len(X[0]), 100, len(Y[0]))
model.train(X_train, Y_train, 0.1, 20)

# Testing the Model
correct = 0
for i in range(len(X_test)):
  model.feed_forward(X_test[i])
  pred = model.out_output
  if np.argmax(pred) == np.argmax(Y_test[i]):
    correct += 1
print("********************* Test Accuracy *********************")
print("Accuracy on test set:", correct/len(X_test))