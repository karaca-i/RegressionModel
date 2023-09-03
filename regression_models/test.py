import numpy as np
import matplotlib.pyplot as plt
from time import time

# def learn_linear0sadas(client_sid, msg):
#     print(msg)
#     arr = msg["data"]
#     y = np.array([i[0] for i in arr])
#     for i in arr:
#         i.pop(0)
#     x = np.array(arr)
#     x = zscore_normalize_features(x)
#     w = np.array([1 for i in arr[0]])
#     b = 1
#     alpha = msg["alpha"]
#     lambd = msg["lambda"]
#     i = 1
#     while True:
#         while client_sid not in running:
#             if client_sid not in clients:
#                 return
#         if client_sid not in clients:
#             return
#         cost = poly_mult.compute_reg_cost(w, b, x, y, lambd)
#         client_data[client_sid] = (i,cost)
#         dj_dw, dj_db = poly_mult.compute_reg_gradient(w,b,x,y,lambd)
#         w = w - alpha * dj_dw
#         b = b - alpha * dj_db
#         i+=1
       
       
       
# APP.PY TESTS

# def learn_linear_old(client_sid, msg):
#     print(msg)
#     arr = msg["data"]

#     y = np.array([i[0] for i in arr])
#     for i in arr:
#         i.pop(0)
        
#     x = np.array(arr)
    
#     if len(arr[0]) <= 1: # single feature
#         w_in = 0.
#         b_in = 100
#         alpha = msg["alpha"]
#         lambd = msg["lambda"]
#         i = 1
        
#         w = w_in
#         b = b_in
#         while True:
#             while client_sid not in running:
#                 if client_sid not in clients:
#                     return
#             if client_sid not in clients:
#                 return
                
#             # finding the model's current state
#             dj_dw, dj_db = lin_single.compute_gradient(w,b,x,y)
#             w = w - alpha * dj_dw
#             b = b - alpha * dj_db
#             f = lin_single.get_model(w,b,x)
            
#             # the total cost at the moment
#             curr_cost = lin_single.compute_cost(w,b,x,y)
            
#             # sending the data to the server
#             client_data[client_sid] = (i,curr_cost)
#             # print(w,b)
#             i+=1     
    
#     else: # multiple features
#         w_in = np.array([0. for i in arr[0]])
#         b_in = 100
#         alpha = msg["alpha"]
#         lambd = msg["lambda"]
#         i = 1
        
#         w = w_in
#         b = b_in
#         while True:
#             while client_sid not in running:
#                 if client_sid not in clients:
#                     return
#             if client_sid not in clients:
#                 return
                
#             # finding the model's current state
#             dj_dw, dj_db = lin_mult.compute_gradient(w,b,x,y)
#             w = w - alpha * dj_dw
#             b = b- alpha * dj_db
#             f = lin_mult.get_model(w,b,x)
            
#             # the total cost at the moment
#             curr_cost = lin_mult.compute_cost(w,b,x,y)
            
#             # sending the data to the server
#             client_data[client_sid] = (i,curr_cost)
#             # print(w,b)
            # i+=1   
            
x = 1
y = np.array([1,2])
print(np.isscalar(y))