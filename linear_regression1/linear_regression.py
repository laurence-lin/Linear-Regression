import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
import math
from scipy import stats as st

'''
Linear regression for ML & data science

'''

# Create random train and test set
#x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size = 0.2, random_state = 1)
dirty_train_set = pd.read_csv('train.csv')
dirty_test_set = pd.read_csv('test.csv')


def preprocessing(x):
    '''
    fillnan,
    x: size [batch, features]
    '''
    x = pd.DataFrame(x).dropna()
    
    return x

train_set = preprocessing(dirty_train_set)
test_set = preprocessing(dirty_test_set)
print('Dirty data size:', dirty_train_set.shape)
print('Clean data size:', train_set.shape)

x_train = train_set['x'].values
y_train = train_set['y'].values

x_test = test_set['x'].values
y_test = test_set['y'].values

# input to model should have shape [batch size, feature]
x_train = x_train.reshape(len(x_train), 1)
y_train = y_train.reshape(len(y_train), 1)
x_test = x_test.reshape(len(x_test), 1)
y_test = y_test.reshape(len(y_test), 1)

print('Mean of x train set:', np.mean(x_train), '\n')
print('Median of x train set:', np.median(x_train), '\n')
print('Mean of y train set', np.mean(y_train), '\n')
print('Median of y train set', np.median(y_train), '\n')
print('Std dev of x train set', np.std(x_train), '\n')
print('Std dev of y train set', np.std(y_train), '\n')

plt.title('Relation between x and y')
plt.scatter(x_train, y_train, color = 'blue', marker = 'x')

plt.subplot(2, 2, 1)
plt.title('X train hist')
plt.hist(x_train)

plt.subplot(2, 2, 2)
plt.title('y train hist')
plt.hist(y_train)

plt.subplot(2, 2, 3)
plt.title('x train')
plt.boxplot(x_train)

plt.subplot(2, 2, 4)
plt.title('y train')
plt.boxplot(y_train)

plt.show()

linear_ = LinearRegression(fit_intercept = True)
linear_.fit(x_train, y_train)

print('R2 score', linear_.score(x_train, y_train))

print('Correlation', math.sqrt(linear_.score(x_train, y_train)))

plt.figure(2)
predict_y = linear_.predict(x_test)
plt.scatter(x_test, y_test, marker = 'x', color = 'red')
plt.plot(x_test, predict_y)
plt.title('Test regression')

plt.show()
    
    
    
    
    
    
    
    
    
    







