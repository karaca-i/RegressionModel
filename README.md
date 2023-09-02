# Regression Model Visualizer
[![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-370/) [![Website](https://img.shields.io/website-up-down-green-red/http/shields.io.svg)](http://regvis.ubien.co/)  
Website: http://regvis.ubien.co/
## **Contributors**
- **İbrahim Karaca**
  <img src="https://raw.githubusercontent.com/ultralytics/assets/main/social/logo-transparent.png" width="2.5%"/>
      <a href="https://github.com/karaca-i">
          <img
            src="https://media.roboflow.com/notebooks/template/icons/purple/github.png?ik-sdk-version=javascript-1.4.3&updatedAt=1672949633691"
            width="2.5%"
          />
      </a>
  <img src="https://raw.githubusercontent.com/ultralytics/assets/main/social/logo-transparent.png" width="2.5%"/>
      <a href="https://www.linkedin.com/in/karaca-ibrahim/">
          <img
            src="https://media.roboflow.com/notebooks/template/icons/purple/linkedin.png?ik-sdk-version=javascript-1.4.3&updatedAt=1672949633691"
            width="2.5%"
          />
      </a>   
      
- **Ahmet Deniz Gelir**
  <img src="https://raw.githubusercontent.com/ultralytics/assets/main/social/logo-transparent.png" width="2.5%"/>
      <a href="https://github.com/adenizgelir0">
          <img
            src="https://media.roboflow.com/notebooks/template/icons/purple/github.png?ik-sdk-version=javascript-1.4.3&updatedAt=1672949633691"
            width="2.5%"
          />
      </a>  

A technical project designed to enable the creation of various types of simple machine learning models.
Once you create a model, you can observe its training performance and effectiveness **in real-time through graphical 
representations.**

The **unique aspect** of this project 
is that <ins>**it doesn't rely on**</ins> external machine learning libraries; instead, all functionalities are implemented from scratch. 
The project **offers control over factors** like learning rate, regularization coefficient, and other 
variables that influence the model's behavior and efficiency.

<h3 align="left">Motivation</h3>  
Discovering insights for inquiries like:  

- How does adjusting the learning rate influence the accuracy of model training?
- How can excessive regularization of model parameters lead to underfitting instead of preventing overfitting?

Or even though you do not understand the basics of machine learning, you can just enjoy investigating the learning process of your self-created models, play around, and predict outputs using your own trained model.

<h3 align="left">Languages and Tools</h3>
<p align="left"> <a href="https://www.python.org" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" alt="python" width="40" height="40"/> </a> <a href="https://developer.mozilla.org/en-US/docs/Web/JavaScript" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/javascript/javascript-original.svg" alt="javascript" width="40" height="40"/> </a> <a href="https://www.w3.org/html/" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/html5/html5-original-wordmark.svg" alt="html5" width="40" height="40"/> </a>  <a href="https://flask.palletsprojects.com/" target="_blank" rel="noreferrer"> <img src="https://www.vectorlogo.zone/logos/pocoo_flask/pocoo_flask-icon.svg" alt="flask" width="40" height="40"/> </a> <a href="https://socket.io/" target="_blank" rel="noreferrer"> <img src="https://www.vectorlogo.zone/logos/socketio/socketio-icon.svg" alt="socketio" width="40" height="40"/> </a> <a href="https://getbootstrap.com" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/bootstrap/bootstrap-plain-wordmark.svg" alt="bootstrap" width="40" height="40"/> </a> <a href="https://www.chartjs.org" target="_blank" rel="noreferrer"> <img src="https://www.chartjs.org/media/logo-title.svg" alt="chartjs" width="40" height="40"/> </a> </p> 

## How to Use 
<p align="center"> <img src="https://github.com/karaca-i/RegressionModel/blob/main/demoresvig.gif" alt="demoGif" width="800" height="450"> </p>  

- **Linear:** The default features, including house size, number of bedrooms, and building age, are initially provided but can be customized by users as they see fit. For example, if you have a dataset of 100 house prices along with their feature information, your model will commence training based on this data, aiming to discover the most suitable model that fits the provided information. Throughout this training process, you can observe your model's progress in real-time, witnessing its continuous improvement.

- **Parameters:** You also have the flexibility to adjust the **alpha** (learning rate) and **lambda** (regularization coefficient) parameters to observe their effects on your model. A lower learning rate may result in slower learning, requiring more time to attain a highly accurate model, while a higher learning rate may lead to issues like gradient descent divergence. It's your task to determine the optimal values for alpha and lambda that yield the best results for your model.  

- **Logistic:** This type of statistical model (also known as logit model) is often used for classification and predictive analytics. Logistic regression estimates the probability of an event occurring, such as voted or didn’t vote, has tumor or not, based on a given dataset of features. Since the outcome is a probability, the dependent variable (y) is bounded between 0 and 1. If the output is 0.8, then it means there is a %80 chance of being "True", depending on your output type.

- **Start/Stop:** You may **pause** the process, inspect, and then keep going where you left off.

- **Graphs:** Upper graph represents the model error (lower is better). Lower graph is your model's representation, if you set your variables wisely, then you should see it gets better every second (gets close to actual data).

- **Predictions:** You may use your **trained model** to predict an output, and see how accurate that prediction is.  

## Models
The functions of all the subsequent models, depending on their feature counts, are derived by:

![single](https://github.com/karaca-i/RegressionModel/blob/main/images/singlemodel.png)

![multi](https://github.com/karaca-i/RegressionModel/blob/main/images/multimodel.png)

### Linear Model
Can be trained using data with one or multiple features, making it a simple and efficient model. However, for intricate problems requiring complex model functions, the accuracy of the trained linear model is often insufficient.  
Various kinds of error functions are utilized within this project, but to illustrate, here is a specific instance:

![error](https://github.com/karaca-i/RegressionModel/blob/main/images/costfunc.png)

### Polynomial Model 
Involves using higher-order functions to fit the model to the user's provided data.
Higher-order variables may need normalization, we have mostly used [Z Score Normalization](https://en.wikipedia.org/wiki/Standard_score)

Furthermore, to mitigate the risk of [overfitting](https://en.wikipedia.org/wiki/Overfitting), we have employed regularization techniques.

![regerror](https://github.com/karaca-i/RegressionModel/blob/main/images/regerror.png)
### Logistic Model 
Used for classification problems, the trained model determines the most suitable category for given features based on the provided data. The classification is influenced by the characteristics of the features themselves.

![sigmoid](https://github.com/karaca-i/RegressionModel/blob/main/images/sigmoid.png)

![logerr](https://github.com/karaca-i/RegressionModel/blob/main/images/logerr.png)

![logregerr](https://github.com/karaca-i/RegressionModel/blob/main/images/logregerr.png)

## Threads
When a user initiates the regression process, the data related to the regression task is transmitted to a central server. Subsequently, a background thread is generated specifically for that user to execute the regression analysis. This background thread is essentially a separate, concurrent process that runs alongside the user's main application.  

The user is kept informed about the progress and status of this background thread at regular intervals, with updates being provided every second. This ensures that the user has real-time visibility into how the regression analysis is proceeding.  

Moreover, the user is granted the capability to initiate and halt the execution of this background thread as needed. This control allows the user to start or stop the regression process at their discretion, providing flexibility in managing the analysis. 

In the event that the user's connection to the server is lost or interrupted (a disconnect occurs), the background thread is terminated automatically. This safeguard prevents any further processing in case of a network problem or loss of communication, ensuring that resources are not wasted on an incomplete or disconnected regression task.  

## Build
Please use `Python 3.6` or higher.  

Install from PyPi:
```bash
pip install -r requirements.txt
```

requirements.txt includes: `numpy`, `flask`, `flask-socketio==5.2.0`, `matplotlib`  
matplotlib is optional, in case you want to test your models. We use chart.js for graphical representations.  

To start the project (developer server):
```bash
python app.py
```
