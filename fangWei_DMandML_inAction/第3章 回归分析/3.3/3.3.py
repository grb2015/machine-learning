# -*- coding: utf-8 -*-
'''
线性回归 根据房子的面积预测售价
官方文档： https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
'''

import matplotlib.pyplot as plt  
import numpy as np  
import pandas as pd  
from sklearn import datasets, linear_model  

######################################################################
'''
    brief   :   读取数据
    input   :   file_name      csv文件的路径
    returns :   X_parameter    [list]      自变量的数组  
                Y_parameter    [list]      因变量的数组  
'''
######################################################################
def get_data(file_name):  
    data = pd.read_csv(file_name)  #用pandas 读取cvs 文件.  
    X_parameter = [] 
    Y_parameter = [] 
    for single_square_feet ,single_price_value in zip(data['square_feet'],data['price']):#遍历数据，  
        X_parameter.append([float(single_square_feet)])#存储在相应的list列表中  
        Y_parameter.append(float(single_price_value))  
    return X_parameter,Y_parameter


######################################################################
'''
    brief   :   线性回归模型的训练  Function for Fitting our data to Linear model  
    input   :   X_parameters    [list]      自变量的数组
                Y_parameters    [list]      因变量的数组
    returns :   regr            [sklearn的LinearRegression对象]    训练好的模型
'''
######################################################################
def linear_model_train(X_parameters,Y_parameters):
    regr = linear_model.LinearRegression()  
    regr.fit(X_parameters, Y_parameters)   #训练模型  
    return regr

# step1 读取数据
X,Y = get_data('input_data.csv')  
# step2 模型训练
regr = linear_model_train(X,Y) 

# step3 预测
predx = 700  #待预测的值 
predX = np.array(predx).reshape(1, -1) # guo modified for origan 必须做成一个二维数组
predY = regr.predict(predX)

# 打印log 
# print('X = ')
# print(X)
# print('Y = ')
# print(Y)
print( "待预测的值 predX = ") 
print(predX)
print( "截距 Intercept value " , regr.intercept_   ) #截距
print( "系数 coefficient" , regr.coef_  ) #系数
print( "预测值 Predicted value: ")
print(predY)

# 绘图
# Function to show the resutls of linear fit model  
def show_linear_line(X_parameters,Y_parameters):  
    regr = linear_model.LinearRegression()  
    regr.fit(X_parameters, Y_parameters)  
    plt.scatter(X_parameters,Y_parameters,color='blue')  
    plt.plot(X_parameters,regr.predict(X_parameters),color='red',linewidth=4)  
    plt.xticks(())  
    plt.yticks(())  
    plt.show()

show_linear_line(X,Y)