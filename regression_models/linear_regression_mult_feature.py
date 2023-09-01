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
    
def get_model(w,b,x):
    normalized_x = zscore_normalize_features(x)
    return np.dot(normalized_x,w)+b

def compute_cost_old(w,b,x,y):
    m,n = x.shape
    return np.sum(np.square(((x @ w.T) + b) - y))/(2*m) 

def compute_cost(w,b,x,y):
    m,n = x.shape
    
    f = get_model(w,b,x)
    err = np.square(f-y)
    total_cost = np.sum(err) / (2*m)
    return total_cost

    
def compute_gradient_old(w,b,x,y):
    m,n = x.shape
    
    dj_dw = np.zeros(n)
    dj_db = 0.

    for i in range(m):
        f = np.dot(w,x[i]) + b
        err = f - y[i]
        for j in range(n):
            dj_dw[j] += err * x[i,j] 
        dj_db += err
        
    dj_dw /= m
    dj_db /= m
    
    return dj_dw, dj_db

def compute_gradient(w,b,x,y):
    m= x.shape[0]
    
    f = get_model(w,b,x)
    err = f - y
    dj_dw = np.dot(x.T,err) / m
    dj_db = np.sum(err) / m
    
    return dj_dw, dj_db

def gradient_decent(w_in,b_in, x,y,alpha,iters):
    m,n = x.shape
    
    w = w_in # nparray
    b = b_in # scalar
    
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

if __name__ == "__main__":
    x_train = np.array([[1,10],[2,12],[4,22],[9,32]]) #size 1.000feet, bedrooms
    y_train = np.array([300,500,900,2100])

    w_in = np.array([0.3,0.2])
    b_in = 100
    bad_model = get_model(w_in, b_in, x_train)
    print(f"actual: {y_train[0]}, bad model: {bad_model[0]}")

    w,b = gradient_decent(w_in, b_in, x_train, y_train, alpha = 4.0e-3, iters = 10000)
    trained_model = get_model(w,b,x_train)
    total_cost = compute_cost(w,b,x_train,y_train)
    print(f"actual: {y_train[0]}, trained model: {trained_model[0]}, cost: {total_cost}")

    x_axis = np.arange(0,4)
    plt.scatter(x_axis,y_train,marker = 'x', c='r',label = "actual prices")
    plt.plot(x_axis,bad_model,label = 'bad model')
    plt.plot(x_axis,trained_model,label ='trained model')
    plt.xlabel("house features")
    plt.ylabel("house prices (100k dollars)")
    plt.legend()
    plt.show()


    
    