import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler


'''
Linear regression for ML & data science

'''

file = pd.read_csv('wine.data')
data = file.values
y_data = data[:, 0]
x_data = data[:, 1:]

shuffle = np.random.permutation(x_data.shape[0])
x_data = x_data[shuffle]
y_data = y_data[shuffle]

total = x_data.shape[0]
train_end = int(total * 0.8)
x_train = x_data[0:train_end, :]
y_train = y_data[0:train_end]
x_test = x_data[train_end:, :]
y_test = y_data[train_end:]  

scaler = StandardScaler() # use standardize to scale the data
scaler.fit(x_train) # compute variance & mean for the scaler object attribute
# Scaling
x_train_std = scaler.transform(x_train)
x_test_std = scaler.transform(x_test)

# Add x0 = 1 constant feature
x_train = np.c_[np.ones((x_train_std.shape[0], 1)), x_train_std]
x_test = np.c_[np.ones((x_test_std.shape[0], 1)), x_test_std]

weight = np.random.random((x_train.shape[1]))

def cost(y, h):
    '''
    mean square error
    '''
    loss = np.mean((y - h)**2) / 2
    
    return loss

def grad(y, h, x):
    '''
    y: size [batch]
    h: size [batch]
    x: size [batch, features]
    return: size [features]
    '''
    grad_w = []
    for feature in range(x.shape[1]):
        grad_i = np.mean((h - y)*x[:, feature])
        grad_w.append(grad_i)
    
    grad_w = np.array(grad_w)
    return grad_w
    
def regression(x, theta):
    '''
    x: size [batch, feature]
    theta: size [batch, feature]
    '''
    h = np.matmul(x, theta)
   
    return h
    
h = regression(x_train, weight)

iterations = 100
learn_rate = 0.3
for iterate in range(iterations):
    
    h = regression(x_train, weight)
    weight -= learn_rate*grad(y_train, h, x_train)
    h = regression(x_train, weight)
    Cost = cost(y_train, h)
    print('Cost:', Cost)

h = regression(x_test, weight)

mse = cost(y_test, h)

print('MSE on test:', mse)
    
    
    
    
    
    
    
    
    
    
    
    







