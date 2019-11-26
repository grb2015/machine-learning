#!/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
if __name__ == "__main__":
    N = 50
    np.random.seed(0)
    print('训练数据集(x,y)：')
    x = np.sort(np.random.uniform(0, 6, N), axis=0)
    print ('x =\n', x)
    y = 2*np.sin(x) + 0.1*np.random.randn(N)
    x = x.reshape(-1, 1)
    print ('y =\n', y)
    print ('SVR - RBF')
    svr_rbf = svm.SVR(kernel='rbf', gamma=0.4, C=100)
    svr_rbf.fit(x, y)
    print ('SVR - Linear')
    svr_linear = svm.SVR(kernel='linear', C=100)
    svr_linear.fit(x, y)
    print ('SVR - Polynomial')
    svr_poly = svm.SVR(kernel='poly', degree=3, C=100)
    svr_poly.fit(x, y)
    print ('Fit OK.')
    x_test = np.linspace(x.min(), 1.5*x.max(), 50)
    print('测试数据集(x_test,y_test)：')
    print('x_test=\n',x_test)
    np.random.seed(0)
    y_test = 2*np.sin(x_test) + 0.1*np.random.randn(N)
    print('y_test=\n',y_test)
    x_test=x_test.reshape(-1,1)
    y_rbf = svr_rbf.predict(x_test)
    y_linear = svr_linear.predict(x_test)
    y_poly = svr_poly.predict(x_test)   
    plt.figure(figsize=(9, 8), facecolor='w')
    plt.plot(x_test, y_rbf, 'r-', linewidth=2, label='RBF Kernel')
    plt.plot(x_test, y_linear, 'g-', linewidth=2, label='Linear Kernel')
    plt.plot(x_test, y_poly, 'b-', linewidth=2, label='Polynomial Kernel')
    plt.plot(x, y, 'ks', markersize=5, label='train data')
    plt.plot(x_test, y_test, 'mo', markersize=6, label='test data')
    plt.scatter(x[svr_rbf.support_], y[svr_rbf.support_], s=200, c='r', marker='*', label='RBF Support Vectors', zorder=10)
    plt.legend(loc='lower left')
    plt.title('SVR', fontsize=16)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.tight_layout(2)
    plt.show()
    print("高斯核函数支持向量机的平均绝对误差为:", mean_absolute_error(y_test,y_rbf))
    print("高斯核函数支持向量机的均方误差为:", mean_squared_error(y_test,y_rbf))
    print("线性核函数支持向量机的平均绝对误差为:", mean_absolute_error(y_test,y_linear))
    print("线性核函数支持向量机的均方误差为:", mean_squared_error(y_test,y_linear))
    print("多项式核函数支持向量机的平均绝对误差为:", mean_absolute_error(y_test,y_poly))
    print("多项式核函数支持向量机的均方误差为:", mean_absolute_error(y_test,y_poly))