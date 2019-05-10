# Building neural network from scratch.
# This version is based on the following blog:
# https://enlight.nyc/projects/neural-network/

# import the required libraries
import numpy as np



# Create the training dataset - input features and labels
X = np.array(([2, 9], [1, 5], [3, 6]), dtype=float)
y = np.array(([92], [86], [89]), dtype=float)


# Scale X and y
X = X/np.amax(X, axis=0)
y = y/100

print(X)
print(y)

# Creating the class for Neural Network

class Neural_Network(object):
    def __init__(self):
        #parameters
        self.inputSize = 2
        self.outputSize = 1
        self.hiddenSize = 3

        #weights
        self.W1 = np.random.randn(self.inputSize, self.hiddenSize)
        self.W2 = np.random.randn(self.hiddenSize, self.outputSize)

    def forward(self, X):
        # forward propagation through the Neural Network
        self.z1 = np.dot(X, self.W1)
        self.z2 = self.sigmoid(self.z1)
        self.z3 = np.dot(self.z2, self.W2)
        output = self.sigmoid(self.z3)

        return output

    def sigmoid(self, s):
        # sigmoid activation function
        return 1/(1+np.exp(-s))

# Create an instance of the class neural network
NN = Neural_Network()

model_output = NN.forward(X)

print("\n Actual output:", str(y))
print("\n Predicted output:", str(model_output))


        

