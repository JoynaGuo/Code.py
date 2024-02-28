#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from scipy.stats import linregress
import glob
for file in glob.glob("C:/Download/*.json"):
    myfile = open(file,"r")
    lines = myfile.readlines()
    for i in lines:
        i=pd.read_json(i)
        del i['summary']['cycle_index'][-1]
        del i['summary']['discharge_capacity'][-1]
        i['summary']['discharge_capacity'][:]=[ k*(1.0/1.1) for k in i['summary']['discharge_capacity']]
        #normalize the discharge capacity 
        plt.plot(i['summary']['cycle_index'],i['summary']['discharge_capacity'])
        i_train, i_test = train_test_split(i, train_size=0.8, test_size=0.2) 
plt.xlabel('Cycle Number')
plt.ylabel('Normalized Discharge Capacity (Ah)')
plt.ylim(0.8,1.2)
plt.xlim(30,2500)


# In[ ]:


for file in glob.glob("C:/Download/*.json"):
    print (file)
    myfile = open(file,"r")
    lines = myfile.readlines()
    for i in lines:
        i=pd.read_json(i)
        i['summary']['discharge_capacity'][:]=[ k*(1.0/1.1) for k in i['summary']['discharge_capacity']]
        plt.plot(i['summary']['cycle_index'][30:***],i['summary']['discharge_capacity'][30:***],linewidth=3)
        sns.scatterplot(i['summary']['cycle_index'][-2:-1], i['summary']['discharge_capacity'][-2:-1]);
        mymodel = numpy.poly1d(numpy.polyfit(i['summary']['cycle_index'][30:***],i['summary']['discharge_capacity'][30:***], 2))
        print(mymodel)
        print(r2_score(i['summary']['discharge_capacity'][30:***], mymodel(i['summary']['cycle_index'][30:***])))
        for j in np.arange((len(i['summary']['cycle_index'])+1),2501):
            i['summary']['cycle_index'].append(j)
        plt.plot(i['summary']['cycle_index'][***:2500],mymodel(i['summary']['cycle_index'][***:2500]),linewidth=1,linestyle='dashed')
        # Define the confidence interval z=1.96 for CI 95%, z=1.64 for CI 90%
        ci = 1.96 *(np.std(mymodel(i['summary']['cycle_index'][***:2500]))) / (np.sqrt(i['summary']['cycle_index'][***:2500]))
        print(ci)
        plt.fill_between(i['summary']['cycle_index'][***:2500], mymodel(i['summary']['cycle_index'][***:2500])-ci,mymodel(i['summary']['cycle_index'][225:2500])+ci, facecolor='yellow', alpha=0.5)
plt.title('Degradation Curve of Lithium Battery')
plt.xlabel('Cycle Number')
plt.ylabel('Discharge Capacity (Ah)')
plt.ylim(0.8,1.0)
plt.xlim(30,2500) 


# In[ ]:


import numpy
from sklearn.metrics import r2_score
plt.plot(FastC19['summary']['cycle_index'][30:-2],FastC19['summary']['discharge_capacity'][30:-2],linewidth=3)
print(FastC19['summary']['cycle_index'][-2],FastC19['summary']['discharge_capacity'][-2])
actual_value=FastC19['summary']['cycle_index'][-2]
print(actual_value)
mymodel = numpy.poly1d(numpy.polyfit(FastC19['summary']['cycle_index'][30:-2],FastC19['summary']['discharge_capacity'][30:-2], 2))
print(mymodel)
print(r2_score(FastC19['summary']['discharge_capacity'][30:-2], mymodel(FastC19['summary']['cycle_index'][30:-2])))
plt.plot(FastC19['summary']['cycle_index'][30:225],FastC19['summary']['discharge_capacity'][30:225],linewidth=3)
mymodel = numpy.poly1d(numpy.polyfit(FastC19['summary']['cycle_index'][30:225],FastC19['summary']['discharge_capacity'][30:225], 2))
print(mymodel)
print(r2_score(FastC19['summary']['discharge_capacity'][30:225], mymodel(FastC19['summary']['cycle_index'][30:225])))
for h in np.arange((len(FastC19['summary']['cycle_index'])+1),2501):
    FastC19['summary']['cycle_index'].append(h)
plt.plot(FastC19['summary']['cycle_index'][225:2500],mymodel(FastC19['summary']['cycle_index'][225:2500]),linewidth=1,linestyle='dashed')
mymodel = numpy.poly1d(numpy.polyfit(FastC19['summary']['cycle_index'][30:400],FastC19['summary']['discharge_capacity'][30:400], 2))
print(mymodel)
print(r2_score(FastC19['summary']['discharge_capacity'][30:400], mymodel(FastC19['summary']['cycle_index'][30:400])))
for h in np.arange((len(FastC19['summary']['cycle_index'])+1),2501):
    FastC19['summary']['cycle_index'].append(h)
plt.plot(FastC19['summary']['cycle_index'][400:2500],mymodel(FastC19['summary']['cycle_index'][400:2500]),linewidth=1,linestyle='dashed')
mymodel = numpy.poly1d(numpy.polyfit(FastC19['summary']['cycle_index'][30:450],FastC19['summary']['discharge_capacity'][30:450], 2))
print(mymodel)
print(r2_score(FastC19['summary']['discharge_capacity'][30:450], mymodel(FastC19['summary']['cycle_index'][30:450])))
for h in np.arange((len(FastC19['summary']['cycle_index'])+1),2501):
    FastC19['summary']['cycle_index'].append(h)
plt.plot(FastC19['summary']['cycle_index'][450:2500],mymodel(FastC19['summary']['cycle_index'][450:2500]),linewidth=1,linestyle='dashed')
import matplotlib.pyplot as plt
import numpy as np
x = np.linspace(0,2000)
mymodel = (-1.011e-06 )*x**2 + (- 1.961e-05)* x + 1.053
plt.plot(x,mymodel,linewidth=1,linestyle='dashed')
x = np.linspace(0,2000)
mymodel = (-7.844e-07)*x**2 + (- 1.961e-05)* x + 1.053
plt.plot(x,mymodel,linewidth=1,linestyle='dashed')

