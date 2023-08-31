# Regression Model Visualizer
A technical project designed to enable the creation of various types of simple machine learning models; 
**linear, polynomial, logistic** (for classification), **multi-featured models**.

Once you create a model, you can observe its training performance and effectiveness **in real-time through graphical 
representations.**

The unique aspect of this project 
is that **it doesn't rely on** external machine learning libraries; instead, all functionalities are implemented from scratch. 
The project offers control over factors like learning rate, regularization coefficient, and other 
variables that influence the model's behavior and efficiency.

<h3 align="left">Languages and Tools:</h3>
<p align="left"> <a href="https://www.python.org" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" alt="python" width="40" height="40"/> </a> <a href="https://developer.mozilla.org/en-US/docs/Web/JavaScript" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/javascript/javascript-original.svg" alt="javascript" width="40" height="40"/> </a> <a href="https://www.w3.org/html/" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/html5/html5-original-wordmark.svg" alt="html5" width="40" height="40"/> </a>  <a href="https://flask.palletsprojects.com/" target="_blank" rel="noreferrer"> <img src="https://www.vectorlogo.zone/logos/pocoo_flask/pocoo_flask-icon.svg" alt="flask" width="40" height="40"/> </a> <a href="https://heroku.com" target="_blank" rel="noreferrer"> <img src="https://www.vectorlogo.zone/logos/heroku/heroku-icon.svg" alt="heroku" width="40" height="40"/> </a>  </p>

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

## Dependencies
Install from PyPi:
```bash
pip install numpy
```

Main motivation behind this project is understand how machine learning models are working behind the scenes and understanding the effects of changing: parameters, learning rate and regularization coefficient lambda of the models by
visualizing by a graph.


