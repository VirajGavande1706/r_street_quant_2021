#!/usr/bin/env python
# coding: utf-8

# In[14]:


import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt


# In[15]:


df=pd.read_csv('train_data.csv')
df
df_y=pd.read_csv('train_target.csv')
df_y


# In[16]:


df=df[['open','high','low','close','SMA_7','SMA_14','SMA_21','RSI_7','RSI_14','RSI_21','ATR_14','bb_mavg','bb_hband','bb_lband']]
df
df_y=df_y[['target']]
df_y


# In[17]:


training_no=0.70*len(df)
training_no=int(training_no)
training_no


# In[18]:


from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from keras.models import Sequential
from keras.layers import Dense, LSTM


# In[19]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,1))
dataset=df.values
scaled_data = scaler.fit_transform(dataset)
scaled_data


# In[51]:


train_data_x = scaled_data[0:training_no, :]
train_data_x.shape
train_data_y=df_y.values[0:training_no,:]
train_data_y.shape


# In[68]:


x_train=[]
y_train=[]
for i in range (25,len(train_data_x)):
    x_train.append(train_data_x[i-25:i,:])
    m=float(np.sum(train_data_y[i-25:i,:])/25)
    if(m<0.5):
        y_train.append(0)
    else:
        y_train.append(1)


# In[69]:


x_train,y_train=np.array(x_train),np.array(y_train)
y_train.shape


# In[70]:


from keras.models import Sequential
from keras.layers import Dense, LSTM


# In[71]:


model=Sequential()
model.add(LSTM(128, return_sequences=True, input_shape= (x_train.shape[1], 14)))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1,activation='sigmoid'))


# In[72]:


model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])


# In[73]:


model.fit(x_train,y_train,batch_size=1,epochs=1)


# In[74]:


x_test=[]
test_data_x=scaled_data[training_no-25:,:]
for i in range (25,len(test_data_x)):
    x_test.append(test_data_x[i-25:i,:])


# In[75]:


x_test=np.array(x_test)
predictions=model.predict(x_test)
predictions


# In[76]:


for i in range (len(predictions)):
    print(predictions[i])


# In[77]:


pred=[]
for i in range(len(predictions)):
    if(predictions[i]<0.3):
        pred.append(0)
    else:
        pred.append(1)
pred=np.array(pred)
for i in range(len(pred)):
    print(pred[i])


# In[78]:


pred.shape


# In[79]:


7650+17850


# In[80]:


y_test=df_y.values[17850:,0]
y_test.shape


# In[81]:


true_p=0
for i in range(0,len(pred)):
    if(pred[i]==y_test[i]):
        true_p=true_p+1
acc=float(true_p/7650)
acc


# In[82]:


x_test1=pd.read_csv('test_data.csv')
x_test1


# In[83]:


x_test1=x_test1[['open','high','low','close','SMA_7','SMA_14','SMA_21','RSI_7','RSI_14','RSI_21','ATR_14','bb_mavg','bb_hband','bb_lband']]
x_test1.shape


# In[84]:


df


# In[85]:


x_test2=df.loc[25475:,:]
x_test2.shape


# In[86]:


x_test2=x_test2.append(x_test1)


# In[87]:


x_test2.shape


# In[88]:


test_dataset=x_test2.values
scaled_test_data = scaler.fit_transform(test_dataset)
scaled_test_data


# In[89]:


x_test_3=[]
for i in range (25,10342):
    x_test_3.append(scaled_test_data[i-25:i,:])
x_test_3=np.array(x_test_3)
x_test_3.shape


# In[90]:


test_predictions=model.predict(x_test_3)
test_predictions


# In[91]:


for i in range(len(test_predictions)):
    print(test_predictions[i])


# In[92]:


test_predictions.shape


# In[93]:


submission2=pd.DataFrame(test_predictions, columns = ['target']).to_csv('submission2.csv')


# In[ ]:




