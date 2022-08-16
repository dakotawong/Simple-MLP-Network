import numpy as np
from utils import *

# Class for the MLP network (3 layers)
class MLP():
  
  # Creates model object
  def __init__(self, n_input, n_hidden, n_output):

    # Weights between input and hidden layers
    self.Input_To_Hidden_W = np.random.randn(n_hidden, n_input + 1) # +1 is for biases

    # Weights between hidden and output layers
    self.Hidden_To_Output_W = np.random.randn(n_output, n_hidden + 1) # +1 is for biases

    # Values that will be used by the model
    self.input = None
    self.in_hidden = None
    self.out_hidden = None  
    self.in_output = None
    self.out_output = None

  # Feeds input through the entire network
  # Feeds input x through the model and updates the model object
  def feed_forward(self, x):
     
     # Input to hidden layer
     self.input = np.append(x, [[-1]], axis = 0) # Appends -1 for the biases
     self.in_hidden = np.dot(self.Input_To_Hidden_W, self.input)
     self.out_hidden = sigmoid(self.in_hidden)

     # Hidden layer to Output
     self.out_hidden = np.append(self.out_hidden, [[-1]], axis = 0) # Appends -1 for the biases
     self.in_output = np.dot(self.Hidden_To_Output_W, self.out_hidden)
     self.out_output = softmax(self.in_output)
     
     return

  # Updates weights using gradient descent for a single training example
  # parameters:
  #     x ----------> input
  #     y ----------> predicted label
  #     target -----> training label
  #     lr ---------> Learning rate
  def back_propagation(self, x, y, target, lr):
     delta_out = (target - y)*softmax_gradient(self.out_output)
     self.Hidden_To_Output_W += lr * np.dot(delta_out, self.out_hidden.T)
     delta_hidden = sigmoid_gradient(self.in_hidden) * np.dot(self.Hidden_To_Output_W[:, :-1].T, delta_out)
     self.Input_To_Hidden_W += lr * np.dot(delta_hidden, self.input.T)
     return

  # Trains the network
  #     X ----------> 2D List of training inputs
  #     Y ----------> 2D List of training labels
  #     lr ---------> Learning rate
  #     e ----------> Number of Epochs 
  def train(self, X, Y, lr, e):
    for epoch in range(e):
      print(f"********************* Epoch {epoch+1} *********************")
      cost = 0
      for i in range(len(X)):
        self.feed_forward(X[i])
        cost += squared_error_loss(Y[i], self.out_output)
        self.back_propagation(X[i], self.out_output, Y[i], lr)
      print("Cost value:", cost)