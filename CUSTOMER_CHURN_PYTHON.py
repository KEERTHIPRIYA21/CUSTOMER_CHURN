#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#Load the data
dataf = pd.read_csv("customer_data.csv")
dataf.sample(5)


# In[3]:


#Keep all columns other than CustomerID and Name
columns_to_keep = ['Age', 'Gender', 'Location','Subscription_Length_Months','Monthly_Bill','Total_Usage_GB','Churn']
dataf = dataf[columns_to_keep]
dataf.sample(5)


# In[4]:


#datatype of attributes
dataf.dtypes


# In[5]:


'''Quick glance at above makes me realize that Gender and Location should be integers but it is an object. Let's check what's going on with this column

Check for null values'''

dataf[pd.to_numeric(dataf.Age,errors='coerce').isnull()]
dataf[pd.to_numeric(dataf.Gender,errors='coerce').isnull()]
dataf[pd.to_numeric(dataf.Location,errors='coerce').isnull()]
dataf[pd.to_numeric(dataf.Subscription_Length_Months,errors='coerce').isnull()]
dataf[pd.to_numeric(dataf.Monthly_Bill,errors='coerce').isnull()]
dataf[pd.to_numeric(dataf.Total_Usage_GB,errors='coerce').isnull()]
dataf[pd.to_numeric(dataf.Churn,errors='coerce').isnull()]

#No null values found


# In[6]:


#VISUALIZATION


# In[7]:


mb_churn_no = dataf[dataf.Churn==0].Monthly_Bill      
mb_churn_yes = dataf[dataf.Churn==1].Monthly_Bill     

plt.xlabel("Monthly Charges")
plt.ylabel("Number Of Customers")
plt.title("Customer Churn Prediction Visualiztion")

plt.hist([mb_churn_yes, mb_churn_no], rwidth=0.95, color=['green','red'],label=['Churn=1','Churn=0'])
plt.legend()


# In[8]:


S_churn_no = dataf[dataf.Churn==0].Subscription_Length_Months     
S_churn_yes = dataf[dataf.Churn==1].Subscription_Length_Months     

plt.xlabel("Subscription_Length_Months")
plt.ylabel("Number Of Customers")
plt.title("Customer Churn Prediction Visualiztion")

plt.hist([S_churn_yes, S_churn_no], rwidth=0.95, color=['green','red'],label=['Churn=1','Churn=0'])
plt.legend()


# In[9]:


dataf.columns = dataf.columns.str.strip()
print(dataf.columns)


# In[10]:


#Many of the columns are yes, no etc. Let's print unique values in object columns to see data values
def print_unique_col_values(dataf):
       for column in dataf:
            if dataf[column].dtypes=='object':
                print(f'{column}: {dataf[column].unique()}') 


# In[11]:


#Print unique values in columns
print_unique_col_values(dataf)


# In[12]:


dataf['Gender'].replace({'Female':1,'Male':0},inplace=True)
dataf.dtypes


# In[16]:


for col in dataf:
    print(f'{col}: {dataf[col].unique()}') 


# In[18]:


dataf = pd.get_dummies(data=dataf, columns=['Location'])
dataf.columns


# In[19]:


dataf.dtypes


# In[20]:


#Normalize or scale attribute values to the range of 0 and 1


# In[21]:


get_ipython().system('pip install scikit-learn')


# In[22]:


cols_to_scale = ['Subscription_Length_Months','Monthly_Bill','Age','Total_Usage_GB']

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
dataf[cols_to_scale] = scaler.fit_transform(dataf[cols_to_scale])


# In[23]:


for col in dataf:
    print(f'{col}: {dataf[col].unique()}')


# In[24]:


X = dataf.drop('Churn',axis='columns')
y = dataf['Churn']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=5)


# In[25]:


X_train.shape


# In[26]:


X_test.shape


# In[27]:


X_train[:10]


# In[28]:


len(X_train.columns)


# In[78]:


#Build a model (ANN) in tensorflow/keras

import tensorflow as tf
from tensorflow import keras


model = keras.Sequential([
    keras.layers.Dense(10, input_shape=(10,), activation='relu'),
    keras.layers.Dense(9, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

# opt = keras.optimizers.Adam(learning_rate=0.01)

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=200)


# In[68]:


model.evaluate(X_test, y_test)


# In[69]:


yp = model.predict(X_test)
yp[:5]


# In[70]:


yp = model.predict(X_test)
yp[:5]


# In[71]:


y_pred = []
for element in yp:
    if element >0.5:
        y_pred.append(1)
    else:
        y_pred.append(0)


# In[74]:


y_pred[:10]


# In[75]:


y_test[:10]


# In[76]:


from sklearn.metrics import confusion_matrix , classification_report

print(classification_report(y_test,y_pred))


# In[77]:


import seaborn as sn
cm = tf.math.confusion_matrix(labels=y_test,predictions=y_pred)

plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')


# In[66]:


y_test.shape


# In[81]:


#Accuracy
round((8400+1630)/(8400+1630+1550+8420),2)


# In[82]:


#Precision for 0 class. i.e. Precision for customers who did not churn

round(8400/(8400+8420),2)


# In[83]:


#Precision for 1 class. i.e. Precision for customers who actually churned
round(1630/(1630+1550),2)


# In[ ]:


#Recall for 0 class


# In[84]:


round(8400/(8400+1550),2)


# In[85]:


round(1630/(1630+8420),2)


# In[ ]:




