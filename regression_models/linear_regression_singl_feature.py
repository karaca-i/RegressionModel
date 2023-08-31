import numpy as np
import matplotlib.pyplot as plt

def get_model_old(w,b,x):
    m = x.shape[0]

    f = np.zeros(m)
    f = w * x + b
    
    return f

def get_model(w,b,x):
    f = np.dot(w,x) + b
    return f 

def compute_cost_old(w,b,x,y):
    m = x.shape[0]
    
    f = get_model(w,b,x)
    
    for i in range(m):
        err = f[i] - y[i]
        err = err **2
        total_cost += err
    
    total_cost /= (2*m)
    return total_cost    

def compute_cost(w,b,x,y):
    m = x.shape[0]
    
    f = get_model(w,b,x)
    err = f - y
    total_cost = np.sum(np.square(err)) / (2*m)
    return total_cost

def compute_gradient_old(w,b,x,y):
    m = x.shape[0]
    
    dj_dw = 0.
    dj_db = 0.
    
    f = get_model(w,b,x)
    
    for i in range(m):
        err = f[i] - y[i]
        dj_dw += err * x[i]
        dj_db += err
        
    dj_dw /= m
    dj_db /= m
    
    return dj_dw, dj_db

def compute_gradient(w,b,x,y):
    m = x.shape[0]
    
    f = get_model(w,b,x)
    err = f - y
    dj_dw = np.sum(np.dot(err,x)) / m
    dj_db = np.sum(err) / m
    
    return dj_dw, dj_db

def gradient_decent(w_in,b_in, x,y,alpha,iters):
    w = w_in
    b = b_in 
    
    for i in range(iters):
        dj_dw, dj_db = compute_gradient(w,b,x,y)
        w -= alpha * dj_dw
        b -= alpha * dj_db
    
    return w,b

def get_generalized_model(w,b,x,y,alpha,iters):
    
    w_new , b_new = gradient_decent(w,b,x,y,alpha,iters)
    
    f = get_model(w_new,b_new,x)
    return f


# testing
x_train = np.array([1.0,2.4,4.0,7.0])
y_train = np.array([300.0,500.0,900.0,2100.0])
plt.scatter(x_train,y_train,marker='x',c='r',label = "House Pricing")

w_in = 0
b_in = 200
bad_model = get_model(w_in,b_in,x_train)
plt.plot(x_train,bad_model,label="bad model")

good_model = get_generalized_model(w_in,b_in,x_train,y_train,alpha = 1.0e-2,iters = 10000)
plt.plot(x_train,good_model,label = "trained model")

plt.legend()
plt.show()

    
    