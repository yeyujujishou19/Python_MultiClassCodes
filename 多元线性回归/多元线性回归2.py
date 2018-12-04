from numpy import genfromtxt
import numpy as np
from sklearn import datasets,linear_model

# 读取数据
dataPath=r"E:\sxl_Programs\Python\多元线性回归\Delivery2.csv"
deliveryData=genfromtxt(dataPath,delimiter=',')

print ("data")
print (deliveryData)

X=deliveryData[:,:-1]  #所有行，第一列到最后一列前（不包括最后一列）
Y=deliveryData[:,-1]   #所有行，最后一列

print ("X:")
print (X)
print ("Y:")
print (Y)

regr=linear_model.LinearRegression()

regr.fit(X,Y)

print ("coefficients")
print (regr.coef_)
print ("intercept:")
print (regr.intercept_)

xPred = [[102,6,0,1,0]]
yPred = regr.predict(xPred)
print("predicted y:")
print (yPred)