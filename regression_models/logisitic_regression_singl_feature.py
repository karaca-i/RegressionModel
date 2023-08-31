import numpy as np
import matplotlib.pyplot as plt
from utils.plot import plot_data
from utils.normalize import zscore_normalize_features

# x1 = size, x2 = age, y = if has tumor or not

def get_model(w,b,x):
    
    m= x.shape[0]
    # x = zscore_normalize_features(x)

    f = np.zeros(m)
    f = w*x + b
   
    return sigmoid(f)
    
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
        

def compute_cost(w,b,x,y):
    m = x.shape[0]

    total_cost = 0.0
    for i in range(m):
        z = w*x[i] + b
        f = sigmoid(z)
        err = -y[i]*np.log(f) - (1-y[i])*np.log(1-f)
        total_cost += err

    total_cost /= m
    return total_cost

def compute_gradient(w,b,x,y):
    m = x.shape[0]
    dj_dw = 0.
    dj_db =0.

    for i in range(m):
        z = w*x[i] + b
        f = sigmoid(z)
        err = f- y[i]
        dj_dw += err * x[i]
        dj_db += err
    
    dj_dw /= m
    dj_db /= m

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
wnew, bnew = gradient_descent(w,b,x_train,y_train,1e-1,10000)
good_model = get_model(wnew,bnew,x_train)
plt.plot(x_train,good_model,label ='good model')
print(wnew,bnew)
plt.legend()
plt.show()