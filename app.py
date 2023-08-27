from flask import Flask
from flask import render_template

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")
@app.route("/linear/")
def linear():
    return render_template("linear.html")
@app.route("/logistic/")
def logistic():
    return render_template("logistic.html")