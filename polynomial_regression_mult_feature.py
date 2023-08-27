import numpy as np
import matplotlib.pyplot as plt

def get_model(w,b,x):
    m,n = x.shape

    f = np.zeros(m)
    for i in range(m):
        f[i] = np.dot(w,x[i]) + b
    
    return f

def compute_cost(w,b,x,y):
    m,n = x.shape
    
    f = get_model(w,b,x)
    
    for i in range(m):
        err = f[i] - y[i]
        err = err **2
        total_cost += err
    
    total_cost /= (2*m)
    return total_cost    

def compute_reg_cost(w,b,x,y,lambda_):
    m,n = x.shape
    
    non_reg_cost = compute_cost(w,b,x,y)
    
    reg_cost = 0
    for i in range(n):
        reg_cost += w[i]**2
        
    reg_cost = (reg_cost * lambda_) / (2*m)
    return non_reg_cost + reg_cost
    
def compute_gradient(w,b,x,y):
    m,n = x.shape
    
    dj_dw = np.zeros(n)
    dj_db = 0.
    
    f = get_model(w,b,x)
    
    for i in range(m):
        err = f[i] - y[i]
        dj_dw += err * x[i]
        dj_db += err
        
    dj_dw /= m
    dj_db /= m
    
    return dj_dw, dj_db

def compute_reg_gradient(w,b,x,y,lambda_):
    m,n = x.shape
    
    dj_dw, dj_db = compute_gradient(w,b,x,y)
    
    for i in range(n):
        dj_dw[i] += (lambda_ / m) * (w[i])
        
    return dj_dw, dj_db


def gradient_decent(w_in,b_in, x,y,alpha,lambda_,iters):
    m,n = x.shape
    
    w = w_in # nparray
    b = b_in # scalar
    
    for i in range(iters):
        dj_dw, dj_db = compute_reg_gradient(w,b,x,y,lambda_)
        w -= alpha * dj_dw
        b -= alpha * dj_db
    
    return w,b

def get_generalized_model(w,b,x,y,alpha,lambda_,iters):
    
    w_new , b_new = gradient_decent(w,b,x,y,alpha,lambda_,iters)
    
    f = get_model(w_new,b_new,x)
    return f

# testing



    
    