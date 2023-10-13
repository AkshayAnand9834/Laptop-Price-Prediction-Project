#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing liabraries 
import numpy as np
import pandas as pd


# In[2]:


#importing dataset
df=pd.read_csv(r"E:\Machine Learning\laptop_data.csv", encoding="latin-1")
df


# In[3]:


df


# # <font color=black>1.data preprocessing and cleaning</font>
# 

# In[4]:


df.duplicated().sum()


# In[5]:


df.isnull().sum()


# In[6]:


df.info()


# In[7]:


#removing unnessary columns from dataset
#droping laptop_ID 
df.drop(columns=['Unnamed: 0'],inplace=True)


# In[8]:


df.head()


# In[9]:


#removing gb from ram column
df['Ram']=df['Ram'].str.replace('GB', ' ')
df.head()


# In[10]:


#removing kg from Weight Column
df['Weight']=df['Weight'].str.replace('kg',' ')
df


# In[11]:


df.info()


# In[12]:


#converting object datatype of Ram and Weight into int and float
df['Ram']=df['Ram'].astype('int32')
df['Weight']=df['Weight'].astype('float32')
df.info()


# # 2.EDA

# In[13]:


import seaborn as sns


# In[14]:


sns.distplot(df['Price'])


# In[15]:


df['Company'].value_counts().plot(kind='bar')


# In[16]:


#average price per company
import matplotlib.pyplot as plt
sns.barplot(x=df['Company'],y=df['Price'])
plt.xticks(rotation='vertical')
plt.show()


# In[17]:


df['TypeName'].value_counts().plot(kind='bar')


# In[18]:


sns.barplot(x=df['TypeName'],y=df['Price'])
plt.xticks(rotation='vertical')
plt.show()


# In[19]:


sns.distplot(df['Inches'])


# In[20]:


sns.scatterplot(x=df['Inches'],y=df['Price'])


# In[21]:


df.info()


# In[ ]:





# In[22]:


df['ScreenResolution'].value_counts()


# In[23]:


df['Touchscreen']=df['ScreenResolution'].apply(lambda x:1 if 'Touchscreen' in x else 0)


# In[24]:


df.sample(5)


# In[25]:


df['Touchscreen'].value_counts().plot(kind='bar')


# In[26]:


sns.barplot(x=df['Touchscreen'],y=df['Price'])


# In[27]:


df['Ips']=df['ScreenResolution'].apply(lambda x:1 if 'IPS' in x else 0)
df.sample(5)


# In[28]:


df['Ips'].value_counts().plot(kind='bar')


# In[29]:


sns.barplot(x=df['Ips'],y=df['Price'])


# In[30]:


#splitting in screeresolution----list
new=df['ScreenResolution'].str.split('x',n=1,expand=True)


# In[ ]:





# In[31]:


df['X_res']=new[0]
df['Y_res']=new[1]


# In[32]:


df.head()


# In[33]:


#x has some problem to resolve it used regexp
df['X_res']=df['X_res'].str.replace(',','').str.findall(r'(\d+\.?\d+)').apply(lambda x:x[0])


# In[34]:


df.head()


# In[35]:


df.info()


# In[36]:


#converting object into int
df['X_res']=df['X_res'].astype('int')
df['Y_res']=df['Y_res'].astype('int')


# In[ ]:





# In[37]:


#finding correlation with Price column
df.corr()['Price']


# In[38]:


df['ppi']=(((df['X_res']**2)+(df['Y_res']**2))**0.5/df['Inches']).astype('float')
df


# In[39]:


df.corr()['Price']
df


# In[40]:


#Droping Screenresolution because is not required now.
df.drop(columns=['ScreenResolution'],inplace=True)
df.head()


# In[41]:


#Also droping Inches X_res and Y_res column because we we have ppi for analysis
df.drop(columns=['X_res','Y_res','Inches'],inplace=True)


# In[42]:


df.head()


# In[43]:


#Cpu column
df['Cpu'].value_counts()


# In[44]:


#Extracting first 3 words 
df['Cpu Name']=df['Cpu'].apply(lambda x:" ".join(x.split()[0:3]))
df.head()


# In[ ]:





# In[45]:


#writing function for Processor
def fetch_processor(text): 
    if text == 'Intel Core i7' or text == 'Intel Core i5' or text == 'Intel Core i3': 
        return text 
    else:
        if text.split()[0] =='Intel': 
            return 'Other Intel Processor' 
        else: return 'AMD Processor'


# In[46]:


df['Cpu brand']=df['Cpu Name'].apply(fetch_processor)
df.head()


# In[47]:


df['Cpu brand'].value_counts().plot(kind='bar')


# In[48]:


sns.barplot(x=df['Cpu brand'],y=df['Price'])
plt.xticks(rotation='vertical')
plt.show()


# In[49]:


#now droping unecessary Cpu and Cpu Name
df.drop(columns=['Cpu','Cpu Name'],inplace=True)


# In[50]:


df.head()


# In[51]:


#Ram 
df['Ram'].value_counts().plot(kind='bar')


# In[52]:


sns.barplot(x=df['Ram'],y=df['Price'])
plt.xticks(rotation='vertical')
plt.show()


# In[53]:


df['Memory'].value_counts()


# In[54]:


df['Memory'] = df['Memory'].astype(str).replace('\.0', '', regex=True)
df["Memory"] =df["Memory"].str.replace('GB', '') 
df["Memory"] = df["Memory"].str.replace('TB', '000') 
new =df["Memory"].str.split("+", n = 1, expand = True) 

df["first"]= new[0]
df["first"]=df["first"].str.strip()

df["second"]= new[1]

df["Layer1HDD"] = df["first"].apply(lambda x: 1 if "HDD" in x else 0) 
df["Layer1SSD"] = df["first"].apply(lambda x: 1 if "SSD" in x else 0)
df["Layer1Hybrid"] = df["first"].apply(lambda x: 1 if "Hybrid" in x else 0) 
df["Layer1Flash_Storage"] =df["first"].apply(lambda x: 1 if "Flash Storage" in x else 0) 

df['first'] =df['first'].str.replace(r'\D', '')

df["second"].fillna("0", inplace = True)

df["Layer2HDD"] =df["second"].apply(lambda x: 1 if "HDD" in x else 0) 
df["Layer2SSD"] = df["second"].apply(lambda x: 1if "SSD" in x else 0) 
df["Layer2Hybrid"] = df["second"].apply(lambda x: 1 if "Hybrid" in x else 0)
df["Layer2Flash_Storage"] = df["second"].apply(lambda x: 1 if "Flash Storage" in x else 0)

df['second']= df['second'].str.replace(r'\D', '')

df['first']=df['first'].astype(int)
df['second']=df['second'].astype(int)

df['HDD']=(df["first"]*df["Layer1HDD"]+df["second"]*df["Layer2HDD"]) 
df["SSD"]=(df["first"]*df["Layer1SSD"]+df["second"]*df["Layer2SSD"]) 
df["Hybrid"]=(df["first"]*df["Layer1Hybrid"]+df["second"]*df["Layer2Hybrid"]) 
df["Flash_Storage"]=(df["first"]*df["Layer1Flash_Storage"]+df["second"]*df["Layer2Flash_Storage"])

df.drop(columns=['first', 'second', 'Layer1HDD', 'Layer1SSD', 'Layer1Hybrid',
'Layer1Flash_Storage', 'Layer2HDD', 'Layer2SSD', 'Layer2Hybrid', 
                 'Layer2Flash_Storage'],inplace=True)


# In[55]:


df.drop(columns=['Memory'],inplace=True)


# In[56]:


df.head()


# In[57]:


df.corr()['Price']


# In[58]:


df.drop(columns=['Hybrid','Flash_Storage'],inplace=True)


# In[59]:


df.head()


# In[60]:


#Gpu Column
df['Gpu'].value_counts()


# In[61]:


df['Gpu brand']=df['Gpu'].apply(lambda x:x.split()[0])


# In[62]:


df.head()


# In[63]:


df['Gpu brand'].value_counts()


# In[64]:


df=df[df['Gpu brand']!='ARM']


# In[65]:


df['Gpu brand'].value_counts()


# In[66]:


sns.barplot(x=df['Gpu brand'],y=df['Price'],estimator=np.median)
plt.xticks(rotation='vertical')
plt.show()


# In[67]:


df.drop(columns=['Gpu'],inplace=True)


# In[68]:


df.head()


# In[69]:


df['OpSys'].value_counts()


# In[70]:


sns.barplot(x=df['OpSys'],y=df['Price'])
plt.xticks(rotation='vertical')
plt.show()


# In[71]:


#writing function for operating system
def cat_os(inp): 
    if inp == 'Windows 10' or inp == 'Windows 7' or inp == 'Windows 10 S':
        return 'Windows' 
    elif inp == 'macOS' or inp == 'Mac OS X': 
        return 'Mac' 
    else: return 'Others/No OS/Linux'


# In[72]:


df['os']=df['OpSys'].apply(cat_os)


# In[73]:


df.head()


# In[74]:


df.drop(columns=['OpSys'],inplace=True)


# In[75]:


df.head()


# In[76]:


sns.barplot(x=df['os'],y=df['Price'])
plt.xticks(rotation='vertical')
plt.show()


# In[77]:


sns.distplot(df['Weight'])


# In[78]:


sns.scatterplot(x=df['Weight'],y=df['Price'])


# In[79]:


df.corr()['Price']


# In[80]:


sns.heatmap(df.corr())


# In[81]:


#target Column converting into normal so it will not effect ml models
sns.distplot(np.log(df['Price']))


# # 3.Applying Models

# In[82]:


X=df.drop(columns=['Price'])
y=np.log(df['Price'])


# In[83]:


X


# In[119]:


y


# In[85]:


#Applying train test split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.15,random_state=2)


# In[86]:


X_train


# In[87]:


#usiing  one encoder for Company,TypeName,Cpu brand,Gpu brand,os
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score,mean_absolute_error


# In[88]:


#importing models 
from sklearn.linear_model import LinearRegression,Ridge,Lasso 
from sklearn.neighbors import KNeighborsRegressor 
from sklearn.tree import DecisionTreeRegressor 
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor
from sklearn.svm import SVR 
from xgboost import XGBRegressor


# # 1.Linear Regression

# In[89]:


step1=ColumnTransformer(transformers=[('con_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])],remainder='passthrough')

step2=LinearRegression()

pipe=Pipeline([('step1',step1),('step2',step2)])

pipe.fit(X_train,y_train)

y_pred=pipe.predict(X_test)
print('R2 Score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))


# # 2.Ridge Regression
# 

# In[90]:


step1=ColumnTransformer(transformers=[('con_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])],remainder='passthrough')

step2=Ridge(alpha=10)

pipe=Pipeline([('step1',step1),('step2',step2)])

pipe.fit(X_train,y_train)

y_pred=pipe.predict(X_test)
print('R2 Score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))


# # 3.Lasso Regression

# In[91]:


step1=ColumnTransformer(transformers=[('con_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])],remainder='passthrough')

step2=Lasso(alpha=0.001)

pipe=Pipeline([('step1',step1),('step2',step2)])
pipe.fit(X_train,y_train)

y_pred=pipe.predict(X_test)
print('R2 Score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))


# # 4.KNN

# In[92]:


step1=ColumnTransformer(transformers=[('con_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])],remainder='passthrough')

step2=KNeighborsRegressor(n_neighbors=3)

pipe=Pipeline([('step1',step1),('step2',step2)])

pipe.fit(X_train,y_train)

y_pred=pipe.predict(X_test)
print('R2 Score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))


# # 5.Decision Tree

# In[93]:


step1=ColumnTransformer(transformers=[('con_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])],remainder='passthrough')

step2=DecisionTreeRegressor(max_depth=8)

pipe=Pipeline([('step1',step1),('step2',step2)])

pipe.fit(X_train,y_train)

y_pred=pipe.predict(X_test)
print('R2 Score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))


# # 6.SVM

# In[94]:


step1=ColumnTransformer(transformers=[('con_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])],remainder='passthrough')

step2=SVR(kernel='rbf',C=10000,epsilon=0.1)

pipe=Pipeline([('step1',step1),('step2',step2)])

pipe.fit(X_train,y_train)

y_pred=pipe.predict(X_test)
print('R2 Score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))


# # 7.Random Forest

# In[95]:


step1=ColumnTransformer(transformers=[('con_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])],remainder='passthrough')

step2=RandomForestRegressor(n_estimators=100,
                           random_state=3,
                           max_samples=0.5,
                           max_features=0.75,
                           max_depth=15)

pipe=Pipeline([('step1',step1),('step2',step2)])

pipe.fit(X_train,y_train)

y_pred=pipe.predict(X_test)
print('R2 Score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))


# In[117]:


np.exp(y_pred)



# # taking input for predicting laptop price 

# In[124]:


# import numpy as np
input_data = np.array(["HP","Notebook", 6, 2.19, 0, 0, 100.454670,"Intel Core i7", 1000, 0,"AMD","Windows"])
# Make a prediction
predicted_log_price = pipe.predict([input_data])

# If you want to convert the predicted log price back to the original scale:
predicted_price = np.exp(predicted_log_price)
print(predicted_price)


# # <font color=purple>Conclusion-Best model for prediction of laptop price is random forest regressor because of 88.73% accuracy</font>

# # 8)Adaboost

# In[96]:


step1=ColumnTransformer(transformers=[('con_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])],remainder='passthrough')

step2=AdaBoostRegressor(n_estimators=15,learning_rate=1.0)

pipe=Pipeline([('step1',step1),('step2',step2)])

pipe.fit(X_train,y_train)

y_pred=pipe.predict(X_test)
print('R2 Score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))




# # 9)Gradient boost

# In[97]:


step1=ColumnTransformer(transformers=[('con_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])],remainder='passthrough')

step2=GradientBoostingRegressor(n_estimators=500)

pipe=Pipeline([('step1',step1),('step2',step2)])

pipe.fit(X_train,y_train)

y_pred=pipe.predict(X_test)
print('R2 Score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))




# # 10)Xgboost

# In[98]:


step1=ColumnTransformer(transformers=[('con_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])],remainder='passthrough')

step2=XGBRegressor(n_estimators=45,max_depth=5,learning_rate=0.5)

pipe=Pipeline([('step1',step1),('step2',step2)])

pipe.fit(X_train,y_train)

y_pred=pipe.predict(X_test)
print('R2 Score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))




# # 11.Voting Regressor

# In[99]:


from sklearn.ensemble import VotingRegressor,StackingRegressor 
step1=ColumnTransformer(transformers=[('con_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])],remainder='passthrough')

rf=RandomForestRegressor(n_estimators=350,random_state=3,max_samples=0.5,max_features=0.75,max_depth=15)
gb=GradientBoostingRegressor(n_estimators=100,max_features=0.5)
xgb=XGBRegressor(n_estimators=25,max_depth=5,learning_rate=0.3)
step2=VotingRegressor([('rf',rf),('gb',gb),('xgb',xgb)],weights=[5,1,1])
pipe=Pipeline([('step1',step1),('step2',step2)])

pipe.fit(X_train,y_train)

y_pred=pipe.predict(X_test)
print('R2 Score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))


# # 12.Stacking 

# In[100]:


from sklearn.ensemble import StackingRegressor 
step1=ColumnTransformer(transformers=[('con_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])],remainder='passthrough')

estimators=[
    ('rf',RandomForestRegressor(n_estimators=350,random_state=3,max_samples=0.5,max_features=0.75,max_depth=15)),
    ('gb',GradientBoostingRegressor(n_estimators=100,max_features=0.5)),
    ('xgb',XGBRegressor(n_estimators=25,max_depth=5,learning_rate=0.3))
]
step2=StackingRegressor(estimators=estimators,final_estimator=Ridge(alpha=100))
pipe=Pipeline([('step1',step1),('step2',step2)])

pipe.fit(X_train,y_train)

y_pred=pipe.predict(X_test)
print('R2 Score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))


# # Exporting the model

# In[114]:


y_pred


# In[ ]:





# In[ ]:





# In[ ]:




