import numpy as np
import matplotlib.pyplot as plt

x = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y = np.array([0, 0, 0, 1, 1, 1])

# x1 = size, x2 = age, y = if has tumor or not

def get_model(w,b,x):
    
    m,n = x.shape
    model = np.zeros(m)
    