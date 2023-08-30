import numpy as np
import matplotlib.pyplot as plt
from utils.plot import plot_data
from utils.normalize import zscore_normalize_features

# x1 = size, x2 = age, y = if has tumor or not

def get_model(w,b,x):
    
    m, n = x.shape
    x = zscore_normalize_features(x)

    f = np.zeros(m)
    for i in range(m):
        z = np.dot(w,x[i]) + b
        f[i] = sigmoid(z)

    return f
    
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def one_variable_model(w,b,x):
    m = x.shape[0]
    x = zscore_normalize_features(x)
    
    f = np.zeros((m,))
    for i in range(m):
        z = w*x[i] + b
        f[i] = sigmoid(z)                            

    return f
        

def compute_cost(w,b,x,y):
    m,n = x.shape

    total_cost = 0.0
    for i in range(m):
        z = np.dot(w,x[i]) + b
        f = sigmoid(z)
        err = -y[i]*np.log(f) - (1-y[i])*np.log(1-f)
        total_cost += err

    total_cost /= m
    return total_cost 

def compute_cost_regularized(w,b,x,y,lambda_):
    m,n = x.shape

    total_cost = compute_cost(w,b,x,y)
    reg_cost = 0.0
    for i in range(n):
        reg_cost += w[i]**2

    reg_cost = reg_cost * lambda_ / (2*m)
    total_cost += reg_cost
    return total_cost

def compute_gradient(w,b,x,y):
    m,n =x.shape
    dj_dw = np.zeros((n,))
    dj_db =0.

    for i in range(m):
        z = np.dot(w*x[i])+b
        f = sigmoid(z)
        err = f- y[i]
        dj_dw += err * x[i]
        dj_db += err
    
    dj_dw /= m
    dj_db /= m

    return dj_dw, dj_db

def compute_reg_gradient(w,b,x,y,lambda_):
    dj_dw, dj_db = compute_gradient(w,b,x,y)
    dj_dw += w * lambda_ / x.shape[0]

    return dj_dw, dj_db

def plot_classification(x_train, y_train, X_train2, y_train2):
    pos = y_train == 1
    neg = y_train == 0

    dlblue = '#0096ff'
    fig,ax = plt.subplots(1,2,figsize=(8,3))

    #plot 1, single variable
    ax[0].scatter(x_train[pos], y_train[pos], marker='x', s=80, c = 'red', label="y=1")
    ax[0].scatter(x_train[neg], y_train[neg], marker='o', s=100, label="y=0", facecolors='none', 
              edgecolors=dlblue,lw=3)

    ax[0].set_ylim(-0.08,1.1)
    ax[0].set_ylabel('y', fontsize=12)
    ax[0].set_xlabel('x', fontsize=12)
    ax[0].set_title('one variable plot')
    ax[0].legend()

    #plot 2, two variables
    plot_data(X_train2, y_train2, ax[1])
    ax[1].axis([0, 4, 0, 4])
    ax[1].set_ylabel('$x_1$', fontsize=12)
    ax[1].set_xlabel('$x_0$', fontsize=12)
    ax[1].set_title('two variable plot')
    ax[1].legend()
    plt.tight_layout()
    plt.show()

x_train = np.array([0., 1, 2, 3, 4, 5])
y_train = np.array([0,  0, 0, 1, 1, 1])
X_train2 = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y_train2 = np.array([0, 0, 0, 1, 1, 1])

# test for 1 variable
plt.scatter(x_train,y_train,marker = 'x', c = 'r', label = 'actual data')

w = 0
b = 5
bad_model = one_variable_model(w, b,x_train)
plt.plot(x_train,bad_model,label = 'bad model')


    
plt.legend()
plt.show()