# myMLPro

> Recording Format
> Record the commit date of subproject
> The simple description of subproject
> Descending order of time.

------

## Linear model for house price prediction

> 27th, March, 2019

Add a linear model application (my data mining hw2), which is a 3 variables and 2 features model. I adopt **BGD and SGD ** method to find optimal parameters of model.

------- 

## Learn Ensemble Model
> 15th, March, 2019 

Simply got some quick tricks such as stacking, blending, subensemble for building an ensemble model by the library **mlens**(pip install mlens).

-------

## MNIST Digit in Kaggle
> 13rd~14th, March, 2019

In this part, I learn: 
1. how to implement a convolutional network model
2. master the loader of dataset in pytorch. 
3. train the model by adam method and learning-rate-schedule. 
4. test the model 

-----

## Titanic in Kaggle 

>  11st~12nd, March, 2019

This is the first time to join competition in Kaggle. I just simply master the main process of making feature engineer and use the stacking method to run some model, such as random forest classifier, adaboost, SVM ,XGBoost and so on. Finally, I got 0.7767 score and ranked 5554/10144.

------

## Monte carlo method. On 11st, March, 2019

With the monte carlo method, I calculate the value of pi/4, single and double integral. 
![](https://raw.githubusercontent.com/JoshuaQYH/blogImage/master/img/20190312001417.png)
![](https://raw.githubusercontent.com/JoshuaQYH/blogImage/master/img/20190312001458.png)
![](https://raw.githubusercontent.com/JoshuaQYH/blogImage/master/img/20190312001518.png)

----

## Predict the house price in Boston with AdaBoost.  
> On 8th, March, 2019

------

## Some tutorials on pytorch's official site. 
> On 5th, March, 2019
-------------


## Linear Model in PM2.5 Prediction

> Date：from 2019-02-26 15：05 to 2019-02-28 00:04

### Introduction

In this project, I'm required to develop a linear model to finish the task -- PM2.5 Prediction. The project's description is from the hw1 in NTU 19-ML course. Click the [link](https://ntumlta2019.github.io/ml-web-hw1/) to get the detail.

### Goal

- master the data processing method with numpy and pandas
- extract the feature and label
- train linear regression model with gradient descent method
- visualize the data and show the correlation.

### Result

I design a linear regression class based on gradient descent method(GD). But it encoutered Nan data in train, it could not fit normal. 
Finaly, I use sklearn.linear_regression to train and predict. Maybe I think I have to check the sklearn source code. 
![](https://raw.githubusercontent.com/JoshuaQYH/blogImage/master/20190228002703.png)
