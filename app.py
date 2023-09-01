from flask import Flask
from flask import render_template,request
from flask_socketio import SocketIO, send, emit
from threading import Thread, Lock

from regression_models import linear_regression_mult_feature as lin_mult
from regression_models import linear_regression_singl_feature as lin_single
from regression_models import logisitic_regression_singl_feature as log_single
from regression_models import logistic_regression_mult_feature as log_mult
from regression_models import polynomial_regression_mult_feature as poly_mult

import numpy as np

app = Flask(__name__) 
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)


clients = []
running = set()

client_data = {}

    
def zscore_normalize_features(X,rtn_ms=False):
    mu     = np.mean(X,axis=0)  
    sigma  = np.std(X,axis=0)
    X_norm = (X - mu)/sigma      

    if rtn_ms:
        return(X_norm, mu, sigma)
    else:
        return(X_norm)

            
def learn_linear(client_sid, msg):
    print(msg)
    arr = msg["data"]

    y = np.array([i[0] for i in arr])
    for i in arr:
        i.pop(0)
        
    x = np.array(arr)
    x_normalized = zscore_normalize_features(x)
    
    isSingle = len(arr[0]) <= 1
    w_in = 0. if isSingle else np.array([0. for i in arr[0]])
        
    b_in = 100
    alpha = msg["alpha"]
    lambd = msg["lambda"]
    i = 1
        
    w = w_in
    b = b_in
    w_cost = w_in
    b_cost = b_in
    while True:
        while client_sid not in running:
            if client_sid not in clients:
                return
        if client_sid not in clients:
            return
            
        # cost state
        if (isSingle):
            dj_dw_cost, dj_db_cost = lin_single.compute_gradient(w_cost,b_cost,x,y)
            curr_cost = lin_single.compute_cost(w_cost,b_cost,x,y)

        else: 
            dj_dw_cost, dj_db_cost = lin_mult.compute_gradient(w_cost,b_cost,x,y)
            curr_cost = lin_mult.compute_cost(w_cost,b_cost,x,y)
            
        # finding the model's current state
        if (isSingle):
            dj_dw, dj_db = lin_single.compute_gradient(w,b,x_normalized,y)
            f = lin_single.get_model(w,b,x_normalized)
        else: 
            dj_dw, dj_db = lin_mult.compute_gradient(w,b,x_normalized,y)
            f = lin_mult.get_model(w,b,x_normalized)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        w_cost = w_cost - alpha*dj_dw_cost
        b_cost = b_cost - alpha*dj_db_cost
        # sending the data to the server
        client_data[client_sid] = (i,curr_cost,list(f))
        # print(w,b)
        i+=1     

def learn_logistic(client_sid, msg):
    print(msg)
    for i in msg["data"]:
        i[0] = i[0]>0
    print(msg)
    return

    arr = msg["data"]

    y = np.array([i[0] for i in arr])
    for i in arr:
        i.pop(0)
        
    x = np.array(arr)
    
    isSingle = len(arr[0]) <= 1
    w_in = 0. if isSingle else np.array([0. for i in arr[0]])
        
    b_in = 1.
    alpha = msg["alpha"]
    lambd = msg["lambda"]
    i = 1
    
    # zipping
    combined = list(zip(x,y))
    combined.sort(key=lambda pair:pair[1])
    x,y = zip(*combined)
    x = np.array(x)
    y = np.array(y)
        
    x_normalized = zscore_normalize_features(x)
    
    w = w_in
    b = b_in
    w_cost = w_in
    b_cost = b_in
    while True:
        while client_sid not in running:
            if client_sid not in clients:
                return
        if client_sid not in clients:
            return
            
        # cost state
        if (isSingle):
            dj_dw_cost, dj_db_cost = log_single.compute_gradient(w_cost,b_cost,x,y)
            curr_cost = log_single.compute_cost(w_cost,b_cost,x,y) 
            # TODO regularized cost add
        else: 
            dj_dw_cost, dj_db_cost = log_mult.compute_reg_gradient(w_cost,b_cost,x,y,lambd)
            curr_cost = log_mult.compute_cost(w_cost,b_cost,x,y)
            
        # finding the model's current state
        if (isSingle):
            dj_dw, dj_db = log_single.compute_gradient(w,b,x_normalized,y)
            f = log_single.get_model(w,b,x_normalized)
        else: 
            dj_dw, dj_db = log_mult.compute_reg_gradient(w,b,x_normalized,y,lambd)
            f = log_mult.get_model(w,b,x_normalized)
        
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        
        w_cost = w_cost - alpha*dj_dw_cost
        b_cost = b_cost - alpha*dj_db_cost
        # sending the data to the server
        client_data[client_sid] = (i,curr_cost)
        # print(w,b)
        i+=1    
    
@app.route("/")
def index():
    return render_template("index.html")
@app.route("/linear/")
def linear():
    return render_template("linear.html")
@app.route("/logistic/")
def logistic():
    return render_template("logistic.html")

@socketio.on('connect')
def handle_connect():
    print('Client {} connected'.format(request.sid))
    clients.append(request.sid)

@socketio.on('disconnect')
def handle_disconnect():
    print('Client {} disconnected'.format(request.sid))
    clients.remove(request.sid)

@socketio.on("get_data")
def get_data():
    sid = request.sid
    if sid in client_data:
        emit("data",[client_data[sid][0],client_data[sid][1],client_data[sid][2]])
    else:
        emit("data", 0)

@socketio.on("learn_linear")
def learn_linstart(msg):
    running.add(request.sid)
    Thread(target=learn_linear, args=(request.sid,msg,)).start()

@socketio.on("learn_logistic")
def learn_logstart(msg):
    running.add(request.sid)
    Thread(target=learn_logistic, args=(request.sid,msg,)).start()
if __name__ == '__main__':
    socketio.run(app)