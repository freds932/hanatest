#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('jupyter nbconvert --to script Hanatest_jupi.ipynb')


# In[ ]:


# TEST WITH 100 LINES OF DATA


# In[120]:


ls


# In[125]:


#test

d = {'col1': [1, 2], 'col2': [3, 4]}  #DICTIONARY
df = pd.DataFrame(data=d)             #Dataframe
df


# In[4]:


import pyhdb
from pylab import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[4]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[424]:


#  CREATE TABLE WITH COLUMN NAMES IN HANA

"""create column table bike_train_col as
(
SELECT COLUMN_NAME FROM SYS.TABLE_COLUMNS WHERE TABLE_NAME = 'bike_train' and SCHEMA_NAME = 'FRED' ORDER BY POSITION 
); 

"""


# In[ ]:


"""
Data Fields

datetime - hourly date + timestamp  
season -  1 = spring, 2 = summer, 3 = fall, 4 = winter 
holiday - whether the day is considered a holiday
workingday - whether the day is neither a weekend nor holiday
weather - 1: Clear, Few clouds, Partly cloudy, Partly cloudy
2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog 
temp - temperature in Celsius
atemp - "feels like" temperature in Celsius
humidity - relative humidity
windspeed - wind speed
casual - number of non-registered user rentals initiated
registered - number of registered user rentals initiated
count - number of total rentals

"""


# In[7]:


# WORKS
#Source: Bike Train

connection = pyhdb.connect('hana4.micropole.loc', 30215, 'FRED', '1h882W28') #
cursor = connection.cursor()
cursor.execute('SELECT TOP 100 *  from "FRED"."bike_train"')

name=cursor.description     # Get columns
name2=[]
for col in range(len(name)):
    name2.append( cursor.description[col][0])
    
#ALTERNATIVE TO MAKE LIST OF COLS: values = ','.join(str(v) for v in df.head(5))

df=pd.DataFrame.from_records(cursor.fetchall(), columns=name2) #,columns=columns; 

print( df.head() , '\n' ,'\n' ,'Dimension: ', df.shape,'\n' )


# In[8]:


(cursor.description)


# In[9]:


df.info()


# In[10]:


#CATEGORICAL VARIABLES

df.select_dtypes(include=['object']).columns  #select the columns of "dtypes 'object' "


# In[11]:


#change object string type to date type
df['datetime']=pd.to_datetime(df['datetime'])


# In[12]:



# Frequency tables for each categorical feature (object type)
for column in df.select_dtypes(include=['object']).columns: #churn.select_dtypes(include=['object']).columns:  # ['State']: 
    display(pd.crosstab(index=df[column], columns='% observations', normalize='columns'))
    
# columns: Values to group by in the columns (here it is the name but can also be col:churn['Churn?'] )
# Normalize by dividing all values by the sum of values over each column (sum(col)=1)


# In[ ]:





# In[13]:


df.columns  #df.columns.tolist()


# In[14]:


#Convert columns to numeric except DATETIME
cols=[i for i in df.columns if i not in ["datetime"]]
for col in cols:
    df[col]=pd.to_numeric(df[col])


# In[15]:


df.dtypes


# In[16]:


type(df.select_dtypes(include=['object']))


# In[17]:


#NUMERIC VARIABLES

df.select_dtypes(exclude=['object']).columns  #select the columns of "dtypes 'object' "


# In[18]:


#(df[['season']])  not work
#df.count
df.iloc[:,11].head()


# In[19]:


#Distri of count is skewed (maybe caused by outliers)

plt.hist(df['count']) #or df.iloc[:,11]
plt.grid(True)
plt.xlabel('Distri Count')
plt.ylabel('Count')
plt.title('Distri Count')
plt.axvline(df['count'].mean(),color='g')
plt.axvline(df['count'].median(),color='r')


# In[20]:


x = np.log1p(100) # log (x+1) reduce also the range of the output
np.expm1(x) # exp(x) - 1


# In[21]:


df.dtypes


# In[22]:


#Drop unecessary variabes: casual and registered

#df.drop([('registered',),('casual',)], axis=1, inplace=True)   # Old
df.drop(['registered','casual'], axis=1,inplace=True)


# In[84]:



# We need to convert datetime to numeric for training.
# Let's extract key features into separate numeric columns
def add_features(df):
    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    df['day'] = df['datetime'].dt.day
    df['dayofweek'] = df['datetime'].dt.dayofweek
    df['hour'] = df['datetime'].dt.hour


# In[24]:


add_features(df)


# In[27]:


# NO LOG TARGET  => OUTLIERS
plt.boxplot([df['count']], labels=['count'])
plt.title('Box Plot - Count')
plt.ylabel('Target')
plt.grid(True)


# In[28]:


# Let's see how the data distribution changes with log1p
# Evenly distributed
plt.boxplot([df['count'].map(np.log1p)], labels=['log1p(count)'])
plt.title('Box Plot - log1p(Count)')
plt.ylabel('Target')
plt.grid(True)


# In[29]:


df['count'] = df['count'].map(np.log1p)

#we use "log" as the distribution of "count" is skewed.

columns2 = ['datetime', 'season', 'holiday', 'workingday', 'weather', 'temp',
       'atemp', 'humidity', 'windspeed' ]


# In[30]:


#plt.hist(df["count"])
type(df)


# In[31]:


#The distribution of count is less skewed

plt.hist(df['count']) #or df.iloc[:,11]
plt.grid(True)
plt.xlabel('Distri Count')
plt.ylabel('Count')
plt.title('Distri Count')
plt.axvline(df['count'].mean(),color='g')
plt.axvline(df['count'].median(),color='r')


# In[32]:


# Training = 70% of the data
# Validation = 30% of the data
# Randomize the datset
np.random.seed(5)
l = list(df.index)
np.random.shuffle(l)
df = df.iloc[l]


# In[33]:


df.dtypes


# In[34]:


#split in train and validation set (used for verifying "training accuracy" and for "optimizing parameters" )
#Test set is used for verifying accuracy of a built-up model
rows = df.shape[0]
train = int(.7 * rows)
test = int(.3 * rows)

#print
rows, train, test


# In[58]:


colinput=list(set(df.columns)-{'count','datetime'})


# In[59]:


colinput


# In[62]:


df[:train].loc[:,colinput].head()


# In[63]:


df[:train].loc[:,['count']].head()


# In[72]:


X_train =df[:train].loc[:,colinput]            #df[:train].iloc[:,1:] # Features: 1st column onwards ; pandas.core.frame.DataFrame
y_train = df.loc[:train,'count'].ravel()  #df[:train].iloc[:,0].ravel() # Target: 0th column ;pandas.core.series.Series


# In[77]:



#numpy.ndarray  ; it doesn't work with "pandas.core.series.Series" #type(df_train.iloc[:,0])#

#validation: (used for verifying "training accuracy" and for "optimizing parameters" )

X_validation = df[train:].loc[:,colinput] 
y_validation = df.loc[:train,'count'].ravel()


# In[86]:


#import test set (validate accuracy of built in model

#  *** df_test ***
cursor.execute('SELECT TOP 100 *  from "FRED"."bike_test"')

name=cursor.description     # Get columns
name2=[]
for col in range(len(name)):
    name2.append( cursor.description[col][0])
df_test=pd.DataFrame.from_records(cursor.fetchall(),columns=name2)

#print(cursor.fetchall())  # [(1,), (9,), (8,), 
#print(np.array(cursor.fetchall()))

print( df_test.head() , '\n' ,'\n' ,'Dimension: ', df_test.shape)


# In[87]:



#change object string type to date type
df_test['datetime']=pd.to_datetime(df_test['datetime'])
# Break down time
add_features(df_test)


# In[88]:


df_test.dtypes


# In[82]:


X_train.dtypes  # WEIRD ORDER


# In[104]:



# XGBoost Training Parameter Reference: 
#   https://github.com/dmlc/xgboost/blob/master/doc/parameter.md
#   https://github.com/dmlc/xgboost/blob/master/doc/parameter.rst
import xgboost as xgb
regressor = xgb.XGBRegressor(max_depth=5,eta=0.1,subsample=0.7,num_round=150) #150 trees with depth of 5.
# !! Use "n_estimators" with XGBRegressor, it silently accepts "num_round=150" but deliver 100 rounds

# Subsampling will occur once in every boosting iteration, good to avoid overfitting by taking not all the DB

# eta: [0,1] reduce weight of the features to avoid overfitting and dominance of one feature on others.

# Gradient Boosting: the trees are trained one after another 
# Each subsequent tree is trained primarily with data that had been incorrectly predicted by previous trees. 


# In[105]:


regressor


# In[114]:


#TRAIN
print(" Train: ",X_train.shape, "Type: " , type(X_train),'\n',"Target: ",y_train.shape,"   Type: " ,type(y_train))


# In[115]:


#VALIDATION
print("Validation: ",X_validation.shape, '\n',"Val_Target: ",y_validation.shape)


# In[116]:


X_train.dtypes


# In[117]:


regressor.fit(X_train,y_train)


# In[118]:


regressor.fit(X_train,y_train, eval_set = [(X_train, y_train), (X_validation, y_validation)])


# In[122]:


df['count'].describe()  #TRAIN DATA SET


# In[126]:


eval_result = regressor.evals_result()


# In[127]:


training_rounds = range(len(eval_result['validation_0']['rmse']))


# In[129]:


plt.scatter(x=training_rounds,y=eval_result['validation_0']['rmse'],label='Training Error')
plt.scatter(x=training_rounds,y=eval_result['validation_1']['rmse'],label='Validation Error')
plt.grid(True)
plt.xlabel('Iteration')
plt.ylabel('RMSE')
plt.title('Training Vs Validation Error')
plt.legend()


# In[130]:


xgb.plot_importance(regressor)


# In[ ]:




