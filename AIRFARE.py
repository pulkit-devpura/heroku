#!/usr/bin/env python
# coding: utf-8

# ##### Phase - 1 Importing Libraries and Dataset 

# In[212]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# In[213]:


train_data = pd.read_csv(r"C:\Users\Pulkit_PC\Desktop\Special Assignment\ML\kanan.csv")
train_data.head()


# In[214]:


train_data_true = pd.read_csv(r"C:\Users\Pulkit_PC\Desktop\Special Assignment\ML\kanan.csv")
train_data_true.head()


# In[215]:


train_data.info()
train_data.dropna(inplace = True)


# In[216]:


train_data.isnull().sum()


# ##### Phase 2- Preprocessing the Data
# ##### The Feature with datatype as string(object) needs to be preprocessed therefore,extracting values from Date of Journey, Arrival Time, and Departure Time

# In[217]:


train_data["Journey_day"] = pd.to_datetime(train_data.Date_of_Journey, format="%d/%m/%Y").dt.day
train_data["Journey_month"] = pd.to_datetime(train_data["Date_of_Journey"], format = "%d/%m/%Y").dt.month
train_data.drop(["Date_of_Journey"], axis = 1, inplace = True)


# In[218]:


# For Departure time 
train_data["Dep_hour"] = pd.to_datetime(train_data["Dep_Time"]).dt.hour
train_data["Dep_min"] = pd.to_datetime(train_data["Dep_Time"]).dt.minute
train_data.drop(["Dep_Time"], axis = 1, inplace = True)


# In[219]:


# For Arrival Time 
train_data["Arrival_hour"] = pd.to_datetime(train_data.Arrival_Time).dt.hour
train_data["Arrival_min"] = pd.to_datetime(train_data.Arrival_Time).dt.minute
train_data.drop(["Arrival_Time"], axis = 1, inplace = True)


# In[220]:


train_data.head()


# In[221]:


#For duration 
duration = list(train_data["Duration"])

for i in range(len(duration)):
    if len(duration[i].split()) != 2:    # Check if duration contains only hour or mins
        if "h" in duration[i]:
            duration[i] = duration[i].strip() + " 0m"   # Adds 0 minute
        else:
            duration[i] = "0h " + duration[i]           # Adds 0 hour

duration_hours = []
duration_mins = []
for i in range(len(duration)):
    duration_hours.append(int(duration[i].split(sep = "h")[0]))   
    duration_mins.append(int(duration[i].split(sep = "m")[0].split()[-1]))   


# In[222]:


# Adding it in new column 

train_data["Duration_hours"] = duration_hours
train_data["Duration_mins"] = duration_mins
train_data.drop(["Duration"], axis = 1, inplace = True)


# In[223]:


train_data.head()


# ###### Phase 3: - Handling Categorical Data using One Hot Encoding

# In[224]:


Airline = train_data[["Airline"]]

Airline = pd.get_dummies(Airline, drop_first= True)

Airline.head()


# In[225]:


Source = train_data[["Source"]]

Source = pd.get_dummies(Source, drop_first= True)

Source.head()


# In[226]:


Destination = train_data[["Destination"]]

Destination = pd.get_dummies(Destination, drop_first = True)

Destination.head()


# In[227]:


data_train_1 = pd.concat([train_data, Airline, Source, Destination], axis = 1)
data_train_1.head()


# In[228]:


data_train_1.drop(["Airline", "Source", "Destination","Additional_Info"], axis = 1, inplace = True)


# In[229]:


data_train_1.head()


# In[230]:


data_train_1.columns


# In[231]:


X = data_train_1.drop('Price',axis=1)
X.head()


# In[232]:


y = data_train_1['Price']
y.head()


# In[258]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[259]:


from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.linear_model import LinearRegression, ElasticNet,LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')


models = [['LinearRegression : ', LinearRegression()],
          ['ElasticNet :', ElasticNet()],
          ['KNeighborsRegressor : ', KNeighborsRegressor()],
          ['DecisionTreeRegressor : ', DecisionTreeRegressor()],
          ['RandomForestRegressor : ', RandomForestRegressor()],
          ['GradientBoostingRegressor : ', GradientBoostingRegressor()],
          ['ExtraTreeRegressor : ', ExtraTreeRegressor()]]


# In[260]:


for name, model in models:
    model=model
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(name, "MAE:",(mean_absolute_error(y_test, predictions)))
    print(name,"MSE:",(mean_squared_error(y_test,predictions)))
    print(name,"RMSE:",(np.sqrt(mean_squared_error(y_test,predictions))))
    print(name,"Score:",(r2_score(y_test,predictions)))
    print("\n")


# In[261]:


algorithms = {
    'RandomForestRegressor' : {
                'model' : RandomForestRegressor(),
                'param' : {
                'n_estimators' : [300, 500, 700, 1000, 2100],
                'max_depth' : [3, 5, 7, 9, 11, 13, 15],
                'max_features' : ["auto", "sqrt", "log2"],
                'min_samples_split' : [2, 4, 6, 8]
        }
    },
    'GradientBoostingRegressor' : {
        'model' : GradientBoostingRegressor(),
        'param' : {
            'learning_rate' : [0.5, 0.8, 0.1, 0.20, 0.25, 0.30],
            'n_estimators' : [300, 500, 700, 1000, 2100],
            'criterion' : ['friedman_mse', 'mse']
        }
    }
}


# In[262]:


score = []
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, RandomizedSearchCV
for name, mp in algorithms.items() :
    rs = RandomizedSearchCV(estimator = mp['model'], param_distributions = mp['param'], cv = 10, n_jobs=-1, verbose=3)
    rs.fit(X_train, y_train)
    score.append({
        'model': name,
        'score' : rs.best_score_,
        'params' : rs.best_params_
    })


# In[263]:


final = pd.DataFrame(score, columns=['model','score', 'params'])
final


# In[264]:


final['params'][0]


# In[265]:


regressor = RandomForestRegressor(n_estimators = 500, min_samples_split = 4, max_features = 'sqrt', max_depth = 13)
regressor.fit(X_train, y_train)
prediction = regressor.predict(X_test)
print('RMSE : {}'.format(np.sqrt(mean_squared_error(y_test, prediction))))


# In[266]:


regressor.score(X_train, y_train), regressor.score(X_test, y_test)


# In[267]:


X_test.head()


# In[268]:


prediction[2]


# In[269]:


data_train_1['Price'][1219]


# In[270]:


print('MAE:', mean_absolute_error(y_test, prediction))
print('MSE:', mean_squared_error(y_test, prediction))
print('RMSE:', np.sqrt(mean_squared_error(y_test, prediction)))


# In[271]:


import pickle
file = open('ticketprice.pkl', 'wb')
pickle.dump(regressor, file)


# In[272]:


import json
columns = {
    'data_columns' : [col.lower() for col in X.columns]
}
with open("columns.json","w") as f:
    f.write(json.dumps(columns))


# In[277]:


train_data_true.columns


# In[278]:


train_data.info()


# In[ ]:




