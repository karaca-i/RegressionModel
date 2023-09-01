import numpy as np
import matplotlib.pyplot as plt

def zscore_normalize_features(X,rtn_ms=False):
    mu     = np.mean(X,axis=0)  
    sigma  = np.std(X,axis=0)
    X_norm = (X - mu)/sigma      

    if rtn_ms:
        return(X_norm, mu, sigma)
    else:
        return(X_norm)
    
def get_model_old(w,b,x):
    m = x.shape[0]

    f = np.zeros(m)
    f = w * x + b
    
    return f

# no need to do normalization for single featured model
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

def gradient_decent(w_in,b_in, x_in,y,alpha,iters):
    w = w_in
    b = b_in 
    
    x = zscore_normalize_features(x_in)
    for i in range(iters):
        dj_dw, dj_db = compute_gradient(w,b,x,y)
        w -= alpha * dj_dw
        b -= alpha * dj_db
    
    return w,b

def get_generalized_model(w,b,x,y,alpha,iters):
    
    w_new , b_new = gradient_decent(w,b,x,y,alpha,iters)
    
    f = get_model(w_new,b_new,x)
    return f


if __name__ == '__main__':
    # testing
    x_train = np.array([1.0,2.4,4.0,7.0])
    y_train = np.array([300.0,500.0,900.0,2100.0])
    plt.scatter(x_train,y_train,marker='x',c='r',label = "House Pricing")

    w_in = 0
    b_in = 200
    bad_model = get_model(w_in,b_in,x_train)
    plt.plot(x_train,bad_model,label="bad model")

    w,b = gradient_decent(w_in,b_in,x_train,y_train,1e-2,10000)
    xx = zscore_normalize_features(x_train)
    good_model = get_model(w,b,xx)
    plt.plot(x_train,good_model,label = "trained model")

    plt.legend()
    plt.show()

    
    