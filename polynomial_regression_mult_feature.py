import numpy as np
import matplotlib.pyplot as plt

def get_model(w,b,x):
    m,n = x.shape

    # feature engineering
    x_modified = np.array([[row[0],row[1]**2, row[2]**3] for row in x])
    f = np.zeros(m)
    
    for i in range(m):
        f[i] = np.dot(x_modified[i],w) + b
    
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
        for j in range(n):
            dj_dw[j] += err * x[i,j]
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

def get_regularized_model(w,b,x,y,alpha,lambda_,iters):
    
    w_new , b_new = gradient_decent(w,b,x,y,alpha,lambda_,iters)
    
    f = get_model(w_new,b_new,x)
    return f

# testing

#x_train = np.array([[1,10,3],[2,12,4],[4,22,3],[9,32,8]]) # size, rooms, restrooms
y_train = np.array([300,500,900,2100,3100,4100,5900,7100,9100,11100,14100,22100])
x_indices = np.arange(0,12)
plt.scatter(x_indices,y_train,marker = 'x',c='r',label="actual prices")

num_houses = 12
num_features = 3

# Define increasing values for each feature
sizes = np.linspace(1, 20, num_houses)  # Increasing sizes from 100 to 300 sq. ft.
bedrooms = np.arange(10, 34, 2)    # Increasing bedroom counts from 1 to 12
bathrooms = np.arange(1, num_houses + 1)   # Increasing bathroom counts from 1 to 12

# Combine the features into a 2D array
x_train2 = np.column_stack((sizes, bedrooms, bathrooms))

w_in = np.array([3.0,2.0,1.0])
b_in = 100.
bad_model = get_model(w_in, b_in, x_train2)
# plt.plot(x_indices, y_train,label="target model")
plt.plot(x_indices,bad_model,label = "bad model")

# Now let's create the trained model
trained_model = get_regularized_model(w_in,b_in,x_train2,y_train,alpha= 1.0e-4,lambda_=0.7, iters = 10000)
plt.plot(x_indices,trained_model, label="trained model")

plt.legend()
plt.show()
    
        




    
    