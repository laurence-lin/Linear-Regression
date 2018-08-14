import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#def Normalized(mat):

# Compute loss function
def Loss(X, Y, theta):
    m = len(Y)
    h_x = np.dot(X, theta)
    return (1/(2*m)) * sum( (h_x - Y)**2 )  # MSE

def gradient(X, Y, theta, alpha, iteration):
    m = len(Y)
    h_x = np.dot(X, theta) # hypothesis
    print(len(theta))
    for iterate in range(iteration):
        for param in range(len(theta)):  # for each updating  parameters
            grad = sum( (h_x - Y) * X[:,param] )
            theta[param] -= alpha*(1/m)*grad

        print('Iteration:  ' + str(iterate + 1) + '  Error: ' + str(Loss(X, Y, theta)))

    return theta

# Input data
data = np.loadtxt('ex1data1.txt', delimiter = ',')
X_0 = data[:,0]
Y = data[:,1]
N = data.shape[0]  # number of samples
plt.figure(1)
plt.plot(X_0,Y,'rx')  # plot data points

theta = [0,0]  # Initialize fitting parameters: length = number of X features
X0 = np.ones((N,2)) # add new axis(features) with value 1
X0[:,1] = X_0
X = X0
# hypothesis = theta0*x0 + theta1*x1, x0 is constant here
print(X)
iteration = 20
alpha = 0.001 # learning rate
print('Initial loss: ', Loss(X, Y, theta))

# Gradient Descent
theta = gradient(X, Y, theta, alpha, iteration)
print(theta)

output = np.dot(X,theta)

plt.plot(X_0, output, 'b-')

plt.show()



