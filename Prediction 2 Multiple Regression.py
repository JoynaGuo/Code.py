#!/usr/bin/env python
# coding: utf-8

# In[2]:


from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn import linear_model
from sklearn.model_selection import train_test_split


# In[5]:


data = pd.read_json('C:\Download\FastCharge_000000_CH19_structure.json')


# In[25]:


len(data['summary']['cycle_index'])


# In[26]:


del data['summary']['temperature_maximum'][0],data['summary']['cycle_index'][0],data['summary']['discharge_capacity'][0]


# In[27]:


del data['summary']['temperature_maximum'][-1],data['summary']['cycle_index'][-1],data['summary']['discharge_capacity'][-1]


# In[28]:


fig = plt.figure(figsize = (16, 9))
ax=fig.add_subplot(111,projection='3d')
n=100
ax.scatter(data['summary']['temperature_maximum'],data['summary']['cycle_index'],data['summary']['discharge_capacity'],color="red")
plt.title("Multivariate Regression:3D")
ax.set_xlabel('temperature', fontweight ='bold')
ax.set_zlabel('discharge_capacity', fontweight ='bold')
ax.set_ylabel('cycle_index', fontweight ='bold')
plt.show()


# In[29]:


from sklearn.linear_model import LinearRegression
X = [list(t)for t in zip(data['summary']['temperature_maximum'],data['summary']['energy_efficiency'],data['summary']['discharge_capacity'])]
y = data['summary']['cycle_index']
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.8,shuffle=False)


# In[30]:


regr = linear_model.LinearRegression()
regr.fit(X_test, y_test)
print(regr.intercept_)
print(regr.coef_)
import statsmodels.api as sm
#fit linear regression model
model = sm.OLS(y, X).fit()
#view model summary
print(model.summary())


# In[31]:


p_values = model.summary2().tables[1]['P>|t|']
p_values


# In[32]:


from sklearn.metrics import mean_squared_error, r2_score
y_pred = regr.predict(X)
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)
print("Mean squared error: ", mse)
from math import sqrt
#root_mean_sq_err
rms = sqrt(mse)
print(rms)
print("R-squared: ", r2)
print(y[-1])
print(y_pred[-1])


# In[36]:


accuracy=y_pred[-1]/y[-1]
print(accuracy)


# In[37]:


# Calculate the adjusted R-squared score
n = len(X)  # number of observations
k = 3  # number of predictors
 
adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - k - 1))
print(adj_r2)


# In[38]:


plt.figure() 
plt.scatter(y, y_pred) 
plt.plot([0, max(y)], [0, max(y)], '--', color='red') 
plt.xlabel('Actual') 
plt.ylabel('Predicted') 
plt.title('Predicted vs Actual Values') 
plt.ylim(0,600)
plt.xlim(0,500)
plt.show()


# In[70]:


#optimization
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso


# In[71]:


from sklearn.linear_model import LinearRegression
X = [list(t)for t in zip(data['summary']['temperature_maximum'],data['summary']['energy_efficiency'],data['summary']['discharge_capacity'])]
y = data['summary']['cycle_index']
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.9,shuffle=False)


# In[72]:


lasso=Lasso(normalize=True)
search=GridSearchCV(estimator=lasso,param_grid={'alpha':np.logspace(-5,5,11)},
                    scoring='neg_mean_squared_error',n_jobs=1,refit=True,cv=10)
search.fit(X_train,y_train)


# In[73]:


print(search.best_params_)
print(abs(search.best_score_))


# In[74]:


lasso=Lasso(normalize=True,alpha= 0.1)
lasso.fit(X_train,y_train)
y_lasso_pred=lasso.predict(X_test)
print (y_lasso_pred[-1])


# In[75]:


plt.figure() 
plt.scatter(y_test, y_lasso_pred) 
plt.plot([0, max(y)], [0, max(y)], '--', color='red') 
plt.xlabel('Actual') 
plt.ylabel('Predicted') 
plt.title('Predicted vs Actual Values') 
plt.ylim(0,600)
plt.xlim(0,500)
plt.show()


# In[81]:


from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
model = make_pipeline(StandardScaler(with_mean=False), Lasso(normalize=True,alpha=3.5938))
print(model)


# In[1]:


from sklearn.linear_model import LinearRegression
X = [list(t)for t in zip(data['summary']['temperature_maximum'],data['summary']['energy_efficiency'])]
y = data['summary']['cycle_index']
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.8, shuffle=False)


# In[83]:


regr = linear_model.LinearRegression()
regr.fit(X_test, y_test)
print(regr.intercept_)
print(regr.coef_)
import statsmodels.api as sm
#fit linear regression model
model = sm.OLS(y, X).fit()
#view model summary
print(model.summary())


# In[84]:


from sklearn.metrics import mean_squared_error, r2_score
y_pred = regr.predict(X)
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)
print("Mean squared error: ", mse)
from math import sqrt
#root_mean_sq_err
rms = sqrt(mse)
print(rms)
print("R-squared: ", r2)
print(y[-1])
print(y_pred[-1])


# In[85]:


plt.figure() 
plt.scatter(y, y_pred) 
plt.plot([0, max(y)], [0, max(y)], '--', color='red') 
plt.xlabel('Actual') 
plt.ylabel('Predicted') 
plt.title('Predicted vs Actual Values') 
plt.ylim(0,500)
plt.xlim(0,500)
plt.show()


# In[1]:


#optimization
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso


# In[6]:


data = pd.read_json('C:\Download\FastCharge_000001_CH16_structure.json')


# In[7]:


del data['summary']['temperature_maximum'][0],data['summary']['cycle_index'][0],data['summary']['discharge_capacity'][0]


# In[8]:


del data['summary']['temperature_maximum'][-1],data['summary']['cycle_index'][-1],data['summary']['discharge_capacity'][-1]


# In[9]:


fig = plt.figure(figsize = (16, 9))
ax=fig.add_subplot(111,projection='3d')
n=100
ax.scatter(data['summary']['temperature_maximum'],data['summary']['cycle_index'],data['summary']['discharge_capacity'],color="red")
plt.title("Multivariate Regression:3D")
ax.set_xlabel('temperature', fontweight ='bold')
ax.set_zlabel('discharge_capacity', fontweight ='bold')
ax.set_ylabel('cycle_index', fontweight ='bold')
plt.show()


# In[10]:


from sklearn.linear_model import LinearRegression
import math
X = [list(t)for t in zip(data['summary']['temperature_maximum'],data['summary']['energy_efficiency'],data['summary']['discharge_capacity'])]
y = data['summary']['cycle_index']
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.8,shuffle=False)


# In[11]:


regr = linear_model.LinearRegression()
regr.fit(X_test, y_test)
print(regr.intercept_)
print(regr.coef_)
import statsmodels.api as sm
#fit linear regression model
model = sm.OLS(y, X).fit()
#view model summary
print(model.summary())


# In[12]:


from sklearn.metrics import mean_squared_error, r2_score
y_pred = regr.predict(X)
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)
print("Mean squared error: ", mse)
from math import sqrt
#root_mean_sq_err
rms = sqrt(mse)
print(rms)
print("R-squared: ", r2)
print(y[-1])
print(y_pred[-1])


# In[13]:


accuracy=1-abs(665-659.875)/665
print(accuracy)


# In[14]:


# Calculate the adjusted R-squared score
n = len(X)  # number of observations
k = 3  # number of predictors
 
adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - k - 1))
print(adj_r2)


# In[50]:


plt.figure() 
plt.scatter(y, y_pred) 
plt.plot([0, max(y)], [0, max(y)], '--', color='red') 
plt.xlabel('Actual') 
plt.ylabel('Predicted') 
plt.title('Predicted vs Actual Values') 
plt.ylim(0,1000)
plt.show()


# In[15]:


from sklearn.linear_model import LinearRegression
import math
X = [list(t)for t in zip(data['summary']['temperature_maximum'],data['summary']['energy_efficiency'])]
y = data['summary']['cycle_index']
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.8,shuffle=False)


# In[16]:


regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)
print(regr.intercept_)
print(regr.coef_)


# In[17]:


from sklearn.metrics import mean_squared_error, r2_score
y_pred = regr.predict(X)
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)
print("Mean squared error: ", mse)
from math import sqrt
#root_mean_sq_err
rms = sqrt(mse)
print(rms)
print("R-squared: ", r2)
print(y[-1])
print(y_pred[-1])


# In[18]:


# Calculate the adjusted R-squared score
n = len(X)  # number of observations
k = 3  # number of predictors
 
adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - k - 1))
print(adj_r2)


# In[19]:


plt.figure() 
plt.scatter(y, y_pred) 
plt.plot([0, max(y)], [0, max(y)], '--', color='red') 
plt.xlabel('Actual') 
plt.ylabel('Predicted') 
plt.title('Predicted vs Actual Values') 
plt.ylim(0,1000)
plt.show()


# In[30]:


data = pd.read_json('C:\Download\FastCharge_000001_CH30_structure.json')


# In[31]:


del data['summary']['temperature_maximum'][0],data['summary']['cycle_index'][0],data['summary']['discharge_capacity'][0],data['summary']['dc_internal_resistance'][0],data['summary']['energy_efficiency'][0]


# In[32]:


del data['summary']['temperature_maximum'][-1],data['summary']['cycle_index'][-1],data['summary']['discharge_capacity'][-1],data['summary']['dc_internal_resistance'][-1],data['summary']['energy_efficiency'][-1]


# In[33]:


fig = plt.figure(figsize = (16, 9))
ax=fig.add_subplot(111,projection='3d')
n=100
ax.scatter(data['summary']['temperature_maximum'],data['summary']['cycle_index'],data['summary']['discharge_capacity'],color="red")
plt.title("Multivariate Regression:3D")
ax.set_xlabel('temperature', fontweight ='bold')
ax.set_zlabel('discharge_capacity', fontweight ='bold')
ax.set_ylabel('cycle_index', fontweight ='bold')
plt.show()


# In[34]:


from sklearn.linear_model import LinearRegression
X = [list(t)for t in zip(data['summary']['temperature_maximum'],data['summary']['discharge_capacity'],data['summary']['energy_efficiency'])]
y = data['summary']['cycle_index']
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.8)


# In[35]:


regr = linear_model.LinearRegression()
regr.fit(X_test, y_test)
print(regr.intercept_)
print(regr.coef_)
import statsmodels.api as sm
#fit linear regression model
model = sm.OLS(y, X).fit()
#view model summary
print(model.summary())


# In[36]:


p_values = model.summary2().tables[1]['P>|t|']
p_values


# In[37]:


from sklearn.metrics import mean_squared_error, r2_score
y_pred = regr.predict(X)
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)
print("Mean squared error: ", mse)
from math import sqrt
#root_mean_sq_err
rms = sqrt(mse)
print(rms)
print("R-squared: ", r2)
print(y[-1])
print(y_pred[-1])


# In[38]:


accuracy=1-abs(771-764.866)/771
print(accuracy)


# In[39]:


# Calculate the adjusted R-squared score
n = len(X)  # number of observations
k = 3  # number of predictors
 
adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - k - 1))
print(adj_r2)


# In[40]:


plt.figure() 
plt.scatter(y, y_pred) 
plt.plot([0, max(y)], [0, max(y)], '--', color='red') 
plt.xlabel('Actual') 
plt.ylabel('Predicted') 
plt.title('Predicted vs Actual Values')
plt.xlim(0,1000)
plt.ylim(0,1000)
plt.show()


# In[26]:


from sklearn.linear_model import LinearRegression
X = [list(t)for t in zip(data['summary']['temperature_maximum'],data['summary']['energy_efficiency'])]
y = data['summary']['cycle_index']
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.8)


# In[27]:


regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)
print(regr.intercept_)
print(regr.coef_)


# In[28]:


from sklearn.metrics import mean_squared_error, r2_score
y_pred = regr.predict(X)
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)
print("Mean squared error: ", mse)
from math import sqrt
#root_mean_sq_err
rms = sqrt(mse)
print(rms)
print("R-squared: ", r2)
print(y[-1])
print(y_pred[-1])


# In[29]:


plt.figure() 
plt.scatter(y, y_pred) 
plt.plot([0, max(y)], [0, max(y)], '--', color='red') 
plt.xlabel('Actual') 
plt.ylabel('Predicted') 
plt.title('Predicted vs Actual Values')
plt.xlim(0,1000)
plt.ylim(0,1000)
plt.show()


# In[94]:


regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)
print(regr.intercept_)
print(regr.coef_)
import statsmodels.api as sm
#fit linear regression model
model = sm.OLS(y, X).fit()
#view model summary
print(model.summary())


# In[98]:


p_values = model.summary2().tables[1]['P>|t|']
p_values


# In[4]:


data = pd.read_json('C:/Download\FastCharge_000001_CH38_structure.json')


# In[5]:


del data['summary']['temperature_maximum'][0],data['summary']['cycle_index'][0],data['summary']['discharge_capacity'][0]


# In[6]:


del data['summary']['temperature_maximum'][-1],data['summary']['cycle_index'][-1],data['summary']['discharge_capacity'][-1]


# In[7]:


fig = plt.figure(figsize = (16, 9))
ax=fig.add_subplot(111,projection='3d')
n=100
ax.scatter(data['summary']['temperature_maximum'],data['summary']['cycle_index'],data['summary']['discharge_capacity'],color="red")
plt.title("Multivariate Regression:3D")
ax.set_xlabel('temperature', fontweight ='bold')
ax.set_zlabel('discharge_capacity', fontweight ='bold')
ax.set_ylabel('cycle_index', fontweight ='bold')
plt.show()


# In[8]:


from sklearn.linear_model import LinearRegression
X = [list(t)for t in zip(data['summary']['temperature_maximum'],data['summary']['discharge_capacity'],data['summary']['energy_efficiency'])]
y = data['summary']['cycle_index']
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.8)


# In[11]:


regr = linear_model.LinearRegression()
regr.fit(X_test, y_test)
print(regr.intercept_)
print(regr.coef_)
import statsmodels.api as sm
#fit linear regression model
model = sm.OLS(y, X).fit()
#view model summary
print(model.summary())


# In[62]:


from sklearn.metrics import mean_squared_error, r2_score
y_pred = regr.predict(X)
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)
print("Mean squared error: ", mse)
from math import sqrt
#root_mean_sq_err
rms = sqrt(mse)
print(rms)
print("R-squared: ", r2)
print(y[-1])
print(y_pred[-1])


# In[63]:


accuracy=1-abs(540-508.793)/540
print(accuracy)


# In[64]:


plt.figure() 
plt.scatter(y, y_pred) 
plt.plot([0, max(y)], [0, max(y)], '--', color='red') 
plt.xlabel('Actual') 
plt.ylabel('Predicted') 
plt.title('Predicted vs Actual Values') 
plt.ylim(0,800)
plt.xlim(0,800)
plt.show()


# In[65]:


from sklearn.linear_model import LinearRegression
X = [list(t)for t in zip(data['summary']['temperature_maximum'],data['summary']['energy_efficiency'])]
y = data['summary']['cycle_index']
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.8)


# In[66]:


regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)
print(regr.intercept_)
print(regr.coef_)


# In[67]:


from sklearn.metrics import mean_squared_error, r2_score
y_pred = regr.predict(X)
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)
print("Mean squared error: ", mse)
from math import sqrt
#root_mean_sq_err
rms = sqrt(mse)
print(rms)
print("R-squared: ", r2)
print(y[-1])
print(y_pred[-1])


# In[68]:


plt.figure() 
plt.scatter(y, y_pred) 
plt.plot([0, max(y)], [0, max(y)], '--', color='red') 
plt.xlabel('Actual') 
plt.ylabel('Predicted') 
plt.title('Predicted vs Actual Values') 
plt.ylim(0,800)
plt.xlim(0,800)
plt.show()


# In[69]:


regr = linear_model.LinearRegression()
regr.fit(X_test, y_test)
print(regr.intercept_)
print(regr.coef_)
import statsmodels.api as sm
#fit linear regression model
model = sm.OLS(y, X).fit()
#view model summary
print(model.summary())

