# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 19:19:33 2021

@author: Soumya PC
"""

#importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#loading the dataset
sal_data= pd.read_csv("Salary_Data.csv")
sal_data.columns
sal_data.head()

#Visualization
plt.hist(sal_data.YearsExperience)
plt.hist(sal_data.Salary)
plt.boxplot(sal_data.YearsExperience)
plt.boxplot(sal_data.Salary)
plt.plot(sal_data.YearsExperience,sal_data.Salary,"ro");plt.xlabel('YearsExperience');plt.ylabel('Salary')

#Correlation coeffiecient
sal_data.Salary.corr(sal_data.YearsExperience)
#0.978 which concludes te data to be good.

#Model building
import statsmodels.formula.api as smf
model=smf.ols('Salary~YearsExperience',data=sal_data).fit()
model.params
model.summary()
#Here R-Squared=0.957, Adj. R sqaured=0.955, P value is 0.00
#Calculating the conf_int for 95%
model.conf_int(0.05)
pred= model.predict(sal_data)
Error=sal_data.Salary-pred
Error
#Visualization for actual value and predicted values
plt.scatter(x=sal_data['YearsExperience'],y=sal_data['Salary'],color='red');plt.plot(sal_data['YearsExperience'],pred,color='black');plt.xlabel('YEARSEXPERIENCE');plt.ylabel('Salary')

#RMSE
from sklearn.metrics import mean_squared_error
from math import sqrt
rmse = sqrt(mean_squared_error(sal_data.Salary,pred))
#Here, RMSE is 55.92

#As the correlation coeff is 0.978 we will transform the features or variables for more accuracy
#log transformation
model1=smf.ols('Salary~np.log(YearsExperience)',data=sal_data).fit()
model1.params
model1.summary()
#Here R-sqaured=0.854,Adj R-squared=0.849, p value is 0.007(Intercept),0.00 for YearsExperience(less than 0.05 for both)
model1.conf_int(0.05)
pred1=model1.predict(sal_data)
pred1
Error1= sal_data.Salary-pred1
Error1
#visualization
plt.scatter(x=sal_data['YearsExperience'],y=sal_data['Salary'],color='green');plt.plot(sal_data['YearsExperience'],pred1,color='black');plt.xlabel('YEARSEXPERIENCE');plt.ylabel('Salary')
#rmse
rmse1= sqrt(mean_squared_error(sal_data.Salary,pred1))
#Here rmse1=103.02

#exponential transformation
model2=smf.ols('np.log(Salary)~YearsExperience',data=sal_data).fit()
model2.params
model2.summary()
#Here R-sqaured=0.93,Adj. R-sqaured=0.93,p value is 0.00
model2.conf_int(0.05)
pred2=model2.predict(sal_data)
pred2
Error2=sal_data.Salary-pred2
#RMSE
rmse2=sqrt(mean_squared_error(sal_data.Salary,pred2))
#rmse2=806.30

#visualization
plt.scatter(x=sal_data['YearsExperience'],y=sal_data['Salary'],color='black');plt.plot(sal_data.YearsExperience,np.exp(pred2),color='green');plt.xlabel('YEARSEXPERIENCE');plt.ylabel('Salary')



#Quadratic model
sal_data['YearsExperience_sq']=sal_data.YearsExperience*sal_data.YearsExperience
model3= smf.ols('Salary~YearsExperience+YearsExperience_sq',data=sal_data).fit()
model3.params
model3.summary()
#Here R-squared=0.957,Ad.R-squared=0.954, p=0.00(Intercept),0(Years_exp),0.915(Years_exp_sq)
conf_int3=model3.conf_int(0.05)
pred3= model3.predict(sal_data)
Eror3= sal_data.Salary-pred3
rmse3=sqrt(mean_squared_error(sal_data.Salary,pred3))
#rmse3=55.90

#visualization
plt.scatter(sal_data.YearsExperience,sal_data.Salary,color='red');plt.plot(sal_data.YearsExperience,pred3,color='blue')

#Conclusion- Here we can observe Quadratic model is the better model as R-sqaured,Adj. Rsquare is highest, rmse is 55.90 which is comparatively lesser than all the models


