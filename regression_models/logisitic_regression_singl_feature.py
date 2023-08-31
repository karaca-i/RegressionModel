import numpy as np
import matplotlib.pyplot as plt
from utils.plot import plot_data
from utils.normalize import zscore_normalize_features
from time import time

# x1 = size, x2 = age, y = if has tumor or not

def get_model(w,b,x):
    # x = zscore_normalize_features(x)
   
    return sigmoid(np.dot(w,x)+b)
    
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_cost(w, b, x, y):
    m = x.shape[0]
    
    z = np.dot(x, w) + b
    f = sigmoid(z)
    
    cost = -np.sum(y * np.log(f) + (1 - y) * np.log(1 - f)) / m
    
    return cost

def compute_gradient(w,b,x,y):
    m = x.shape[0]
    
    z = np.dot(w,x) + b
    f = sigmoid(z)
    err = f- y
    dj_dw = np.dot(err, x) / m
    dj_db = np.sum(err) / m

    return dj_dw, dj_db

def gradient_descent(w,b,x,y,alpha,iters):
    m = x.shape[0]
    
    w_new = w
    b_new = b
    
    for i in range(iters):
        dj_dw, dj_db = compute_gradient(w_new,b_new,x,y)
        w_new -= alpha * dj_dw
        b_new -= alpha * dj_db
        
    return w_new,b_new

# x_train = np.array([0., 1, 2, 3, 4, 5])
# y_train = np.array([0,  0, 0, 1, 1, 1])

x_train = np.array([0., 1, 2, 3, 4, 5, 6,7,8,9,10,11,12,13,14,15,16,17,18,19])
y_train = np.array([0,  0, 0, 0, 0, 0,0,0,0,1,1,1,1,1,1,1,1,1,1,1])

X_train2 = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y_train2 = np.array([0, 0, 0, 1, 1, 1])

# test for 1 variable
plt.scatter(x_train,y_train,marker = 'x', c = 'r', label = 'actual data')

w = 0
b = 1
bad_model = get_model(w, b,x_train)
plt.plot(x_train,bad_model,label = 'bad model')
start = time()
wnew, bnew = gradient_descent(w,b,x_train,y_train,1e-1,10000)
good_model = get_model(wnew,bnew,x_train)
end = time()
print(end-start)
plt.plot(x_train,good_model,label ='good model')
plt.legend()
plt.show()