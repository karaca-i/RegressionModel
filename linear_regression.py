import numpy as np
import matplotlib.pyplot as plt

def get_model(w,b,x):
    m,n = x.shape
    
    f = np.zeros(m)
    for i in range(m):
        f[i] = np.dot(w,x[i]) + b
    
    return f


    