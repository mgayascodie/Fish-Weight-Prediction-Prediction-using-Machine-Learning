#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


# In[2]:


from IPython import get_ipython


# In[3]:


data = pd.read_csv("fish.csv")
data


# In[4]:


data.shape


# In[5]:


data.head()


# In[6]:


data.tail()


# In[7]:


data.describe()


# In[8]:


data.nunique()


# In[9]:


data.columns


# In[10]:


data.info()


# In[11]:


data.isnull().sum()


# In[12]:


data["Species"].unique()


# In[13]:


data["Species"].value_counts()


# In[ ]:





# In[14]:


plt.figure(figsize=(15,6))
sns.countplot('Species', data = data, palette='hls')
plt.xticks(rotation = 90)
plt.show()


# In[15]:


sns.pairplot(data)


# In[16]:


data.corr()


# In[17]:


plt.figure(figsize=(15,6))
sns.heatmap(data.corr(), annot = True)
plt.show()


# In[18]:


plt.figure(figsize=(14,5))
sns.boxplot(data['Width'])
plt.xticks(rotation = 90)
plt.show()


# In[19]:


plt.figure(figsize=(15,6))
sns.boxplot(data['Height'])
plt.xticks(rotation = 90)
plt.show()


# In[20]:


plt.figure(figsize=(15,6))
sns.boxplot(data['Length3'])
plt.xticks(rotation = 90)
plt.show()


# In[21]:


plt.figure(figsize=(15,6))
sns.boxplot(data['Length2'])
plt.xticks(rotation = 90)
plt.show()


# In[22]:


plt.figure(figsize=(15,6))
sns.boxplot(data['Length1'])
plt.xticks(rotation = 90)
plt.show()


# In[23]:


plt.figure(figsize=(15,6))
sns.boxplot(data['Weight'])
plt.xticks(rotation = 90)
plt.show()


# In[24]:


fish_Length3 = data['Length3']
Q3 = fish_Length3.quantile(0.75)
Q1 = fish_Length3.quantile(0.25)
IQR = Q3-Q1
lower_limit = Q1 -(1.5*IQR)
upper_limit = Q3 +(1.5*IQR)
length3_outliers = fish_Length3[(fish_Length3 <lower_limit) | (fish_Length3 >upper_limit)]
length3_outliers


# In[25]:


data[142:145]


# In[26]:


data_new = data.drop([142,143,145])
data_new


# In[27]:


data_new.head()


# In[28]:


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  LabelEncoder
from sklearn.ensemble import RandomForestRegressor


# In[29]:


scaler = StandardScaler()


# In[30]:


columns_scaling = ['Weight', 'Length1','Length2','Length3','Height','Width']
data_new[columns_scaling] = scaler.fit_transform(data_new[columns_scaling])
data_new.describe()


# In[31]:


data_cleaned = data_new.drop("Weight", axis=1)
y = data_new['Weight']
y


# In[32]:


x_train, x_test, y_train, y_test = train_test_split(data_cleaned,y, 
                                                    test_size=0.3, 
                                                    random_state=42)


# In[33]:


label_encoder = LabelEncoder()


# In[34]:


x_train['Species'] = label_encoder.fit_transform(x_train['Species'].values)
x_test['Species'] = label_encoder.transform(x_test['Species'].values)


# In[35]:


model = RandomForestRegressor() 
model.fit(x_train, y_train)


# In[36]:


y_pred = model.predict(x_test)


# In[37]:


print("Training Accuracy :", model.score(x_train, y_train))
print("Testing Accuracy :", model.score(x_test, y_test))


# In[38]:


pip install xgboost


# In[ ]:





# In[39]:


import xgboost as xgb
# xgb = XGBRegressor()
xgb1 = xgb.XGBRegressor()


# In[40]:


xgb1.fit(x_train, y_train)
xgb_pred = xgb1.predict(x_test)


# In[41]:


print("Training Accuracy :", xgb1.score(x_train, y_train))
print("Testing Accuracy :", xgb1.score(x_test, y_test))


# In[42]:


xgb1.save_model("model.json")


# In[43]:


pip install streamlit


# In[44]:


import streamlit as st


# In[45]:


st.header("Fish Weight Prediction App")
st.text_input("Enter your Name: ", key="name")


# In[46]:


np.save('classes.npy', label_encoder.classes_)


# In[47]:


label_encoder.classes_ = np.load('classes.npy',allow_pickle=True)


# In[48]:


xgb_best = xgb.XGBRegressor()


# In[49]:


xgb_best.load_model("model.json")


# In[50]:


if st.checkbox('Show Training Dataframe'):
    data


# In[51]:


st.subheader("Please select relevant features of your fish!")
left_column, right_column = st.columns(2)
with left_column:
    inp_species = st.radio('Name of the fish:',
                           np.unique(data['Species']))


# In[52]:


input_Length1 = st.slider('Vertical length(cm)', 0.0, max(data["Length1"]), 1.0)
input_Length2 = st.slider('Diagonal length(cm)', 0.0, max(data["Length2"]), 1.0)
input_Length3 = st.slider('Cross length(cm)', 0.0, max(data["Length3"]), 1.0)
input_Height = st.slider('Height(cm)', 0.0, max(data["Height"]), 1.0)
input_Width = st.slider('Diagonal width(cm)', 0.0, max(data["Width"]), 1.0)


# In[53]:


if st.button('Make Prediction'):
    input_species = label_encoder.transform(np.expand_dims(inp_species, -1))
    inputs = np.expand_dims(
        [int(input_species), input_Length1, input_Length2, input_Length3, input_Height, input_Width], 0)
    prediction = xgb_best.predict(inputs)
    print("final pred", np.squeeze(prediction, -1))
    st.write(f"Your fish weight is: {np.squeeze(prediction, -1):.2f}g")

