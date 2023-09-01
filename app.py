from flask import Flask
from flask import render_template,request
from flask_socketio import SocketIO, send, emit
from threading import Thread, Lock
from regression_models import polynomial_regression_mult_feature as pr
import numpy as np

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)


clients = []
running = set()

client_data = {}

def learn_linear(client_sid, msg):
    print(msg)
    arr = msg["data"]
    y = np.array([i[0] for i in arr])
    for i in arr:
        i.pop(0)
    x = np.array(arr)
    x = pr.zscore_normalize_features(x)
    w = np.array([1 for i in arr[0]])
    b = 1
    alpha = msg["alpha"]
    lambd = msg["lambda"]
    i = 1
    while True:
        while client_sid not in running:
            if client_sid not in clients:
                return
        if client_sid not in clients:
            return
        cost = pr.compute_reg_cost(w, b, x, y, lambd)
        client_data[client_sid] = (i,cost)
        dj_dw, dj_db = pr.compute_reg_gradient(w,b,x,y,lambd)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
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
        emit("data",[client_data[sid][0],client_data[sid][1]])
    else:
        emit("data", 0)

@socketio.on("learn")
def learn_start(msg):
    running.add(request.sid)
    Thread(target=learn_linear, args=(request.sid,msg,)).start()
if __name__ == '__main__':
    socketio.run(app)