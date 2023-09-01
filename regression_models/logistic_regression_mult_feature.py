import numpy as np
import matplotlib.pyplot as plt
# from utils.plot import plot_data
# from utils.plot import plot_classification

# x1 = size, x2 = age, y = if has tumor or not
def zscore_normalize_features(X,rtn_ms=False):
    mu     = np.mean(X,axis=0)  
    sigma  = np.std(X,axis=0)
    X_norm = (X - mu)/sigma      

    if rtn_ms:
        return(X_norm, mu, sigma)
    else:
        return(X_norm)
    
def get_model_old(w,b,x):
    m, n = x.shape

    f = np.zeros(m)
    for i in range(m):
        z = np.dot(w,x[i]) + b
        f[i] = sigmoid(z)

    return f

def get_model(w,b,x): 
    m, n = x.shape

    z = np.dot(x,w) + b
    f = sigmoid(z)
    return f
    
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
        

def compute_cost_old(w,b,x,y):
    m,n = x.shape

    total_cost = 0.0
    for i in range(m):
        z = np.dot(w,x[i]) + b
        f = sigmoid(z)
        err = -y[i]*np.log(f) - (1-y[i])*np.log(1-f)
        total_cost += err

    total_cost /= m
    return total_cost 

def compute_cost(w,b,x,y):
    
    m = x.shape[0]
    z = np.dot(x,w) + b
    f = sigmoid(z)
    cost = 0.
    for i in range(m):
        cost += -y[i] * np.log(f[i]) - (1-y[i])*np.log(1-f[i])
        
    return cost / m

    # vectorized total_cost = -np.sum(y * np.log(f) + (1 - y) * np.log(1 - f)) / m

def compute_cost_regularized_old(w,b,x,y,lambda_):
    m,n = x.shape

    total_cost = compute_cost(w,b,x,y)
    reg_cost = 0.0
    for i in range(n):
        reg_cost += w[i]**2

    reg_cost = reg_cost * lambda_ / (2*m)
    total_cost += reg_cost
    return total_cost

def compute_cost_regularized(w,b,x,y,lambda_):
    m,n = x.shape

    total_cost = compute_cost(w,b,x,y)
    
    reg_cost = np.sum(w**2) * lambda_ / (2*m)
    return total_cost + reg_cost

def compute_gradient_old(w,b,x,y):
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

def compute_gradient(w,b,x,y):
    m = x.shape[0]
    
    f = get_model(w,b,x)
    err = f- y
    dj_dw = np.dot(x.T, err) / m
    dj_db = np.sum(err) / m
    
    return dj_dw, dj_db

def compute_reg_gradient(w,b,x,y,lambda_):
    dj_dw, dj_db = compute_gradient(w,b,x,y)
    dj_dw += np.dot(lambda_,w) / x.shape[0]

    return dj_dw, dj_db

def gradient_descent(w,b,x_in,y,alpha,lambda_,iters):
    m,n = x_in.shape
    
    x = zscore_normalize_features(x_in)
    w_new = w
    b_new = b
    
    for i in range(iters):
        dj_dw, dj_db = compute_reg_gradient(w_new,b_new,x,y,lambda_)
        w_new -= alpha * dj_dw
        b_new -= alpha * dj_db
        
    return w_new,b_new

if __name__ == '__main__':
    x_train2 = np.array([0., 1, 2, 3, 4, 5])
    y_train2 = np.array([0,  0, 0, 1, 1, 1])
    x_train = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5],[0.5, 1.4],[0.8, 1.1]])
    y_train = np.array([0, 0, 0, 1, 1, 1,0,0])

    combined = list(zip(x_train,y_train))
    combined.sort(key=lambda pair:pair[1])
    x_train, y_train = zip(*combined)
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    
    w = np.array([0.,0.2])
    b = 1.
    
    x_indices = np.arange(0,len(y_train))
    bad_model = get_model(w,b,x_train)
    plt.scatter(x_indices,y_train,marker='x',c='r',label = 'actual prices')
    plt.plot(x_indices,bad_model,label = 'bad model')
    
    xx = zscore_normalize_features(x_train)
    wn,bn = gradient_descent(w,b,x_train,y_train,1e-1,0,10000)
    good_model = get_model(wn,bn,xx)
    plt.plot(x_indices,good_model,label = 'trained model')
    # plot_classification(x_train,y_train,x_train2,y_train2)
    # plt.plot(x_train,bad_model,label = 'bad model')

    oldcost = compute_cost(w,b,x_train,y_train)
    print(oldcost)
    cost1 = compute_cost(wn,bn,x_train,y_train)
    print(cost1)
    cost2 = compute_cost_regularized(wn,bn,x_train,y_train,0.4)
    print(cost2)
    plt.legend()
    plt.show()
