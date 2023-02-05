#!/usr/bin/env python
# coding: utf-8

# ## Bike Buyers Prediction
# 

# ### Importing important Packages

# In[2]:


import pandas as pd #data manipulation and analysis
import numpy as np #provides a high-performance multidimensional array object
import math
import seaborn as sns #data visualization
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# In[3]:


#importing the datasets
dataset=pd.read_csv("bike_buyers.csv")


# In[4]:


#reading the csv file
dataset


# In[5]:


dataset.shape


# In[6]:


#reading first five rows
dataset.head()


# In[5]:


#finding total rows and columns in dataset
dataset.shape


# In[6]:


#Generating attributes information: Type info and total null value infomration
#provides the summary of the dataset
dataset.info()


# In[7]:


#total columns in datasets
dataset.columns


# In[7]:


#dataset type
dataset.dtypes


# In[9]:


#displaying all the values of dataset in array
dataset.values


# In[10]:


#finding the unique value
dataset.Occupation.unique()


# In[11]:


dataset.Age.unique()


# In[12]:


# to get the mean, median, quartiles, std, min, max and many more value
#to get descriptive the statistical summary
dataset.describe()


# In[8]:


# finding the missing value using isna() function and return its sum 
dataset.isna().sum().sort_values(ascending=True)


# In[14]:


#creating the graph for showing the missing value on my datasets
missing_value=dataset.isna().sum().sort_values(ascending=True)/len(dataset)*100
missing_value[missing_value!=0].plot(kind="bar",color="brown")


# In[15]:


#filtering the string 
string_columns = dataset.select_dtypes(include='object').columns
print(string_columns)


# In[16]:


#filtering the numerical columns
numeric_columns = dataset.select_dtypes(include=['int64','float64']).columns
print(numeric_columns)


# ## Checking the outliers 

# In[17]:


# 1. Income Outliers
boxplot= dataset.boxplot(column=['Income'],figsize=(3,4),rot=90,fontsize='8',grid=False)


# In[18]:


# to remove the outliers from Income 
income=dataset['Income'].quantile([0.25,0.5,0.75])
threshold=income[0.75]+(1.5*(income[0.75]-income[0.25]))
dataset_copy=dataset.copy()
dataset_copy['Income']=np.where(dataset_copy['Income']>threshold,threshold,dataset_copy['Income'])
boxplot=dataset_copy.boxplot(column=['Income'],figsize=(3,4),rot=90, fontsize='8',grid=False)


# ## Data Cleaning

# In[19]:


dataset.describe()


# In[20]:


#checking the null value
dataset.isna().sum()


# In[21]:


#checking and replacing the Nan values in each column
dataset
copy_dataset=dataset


# In[22]:


#1. Gender: checking nan values
nan_gender=copy_dataset[copy_dataset['Gender'].isnull()]
nan_gender


# In[23]:


#to replace the nan value of gender we have
replace=copy_dataset['Gender'].mode()
replace


# In[24]:


if (str(replace)=='0   Male\ndtype: object'):
    copy_dataset['Gender']=copy_dataset['Gender'].fillna('Male')
else:
    copy_dataset['Gender']=copy_dataset['Gender'].fillna('Female')


# In[25]:


copy_dataset.head()


# In[26]:


#checking the null value after replacing the nan of gender feature
copy_dataset.isna().sum()


# In[27]:


#the Gender have now 0 nan value
copy_dataset['Gender'].isnull().sum()


# In[28]:


# 2. checking nan value for Cars Feature
nan_cars=copy_dataset[copy_dataset['Cars'].isnull()]
nan_cars


# In[29]:


#now replacing those nan of Cars Feature from the datasets
# calculating the median value because to get the integer 
# value since car cannot be counted in float
median_car=copy_dataset['Cars'].median()
copy_dataset['Cars']=copy_dataset['Cars'].fillna(median_car)


# In[30]:


#checking nan are replaced or not in Cars
copy_dataset['Cars'].isnull().sum()


# In[31]:


# 3. checking nan from Childern 
nan_childern=copy_dataset[copy_dataset["Children"].isnull()]
nan_childern


# In[32]:


# to replace in Childern, the median is required to get the int value
median_children=copy_dataset['Children'].median()
copy_dataset['Children']=copy_dataset['Children'].fillna(median_children)


# In[33]:


#checking nan replaced by the medain or not
copy_dataset.loc[805]


# In[34]:


# 4. checking for Age Fetaure
nan_age=copy_dataset[copy_dataset["Age"].isnull()]
nan_age


# In[35]:


# calculating the median to get the int value since age cant be float
median_age=copy_dataset['Age'].median()
copy_dataset['Age']=copy_dataset['Age'].fillna(median_age)


# In[36]:


#checking the result
copy_dataset.loc[225]


# In[37]:


# 5. checking for Income Fetaure
nan_income=copy_dataset[copy_dataset["Income"].isnull()]
nan_income


# In[38]:


# to replace in nan, the mean is calucated
mean_income=copy_dataset['Income'].mean()
copy_dataset['Income']=copy_dataset['Income'].fillna(mean_income)


# In[39]:


#to test the result
copy_dataset.loc[110]


# In[40]:


# 6. checking for Marital Status Fetaure
nan_marital=copy_dataset[copy_dataset["Marital Status"].isnull()]
nan_marital


# In[41]:


# renaming the two words column name 
copy_dataset.rename(columns={'Marital Status':'Marital_Status','Home Owner':'Home_Owner','Purchased Bike':'Purchased_Bike','Commute Distance':'Commute_Distance'},inplace=True)


# In[42]:


#replacing the nan value of Marital Status by calculating mode
mode_marital=copy_dataset['Marital_Status'].mode()
copy_dataset['Marital_Status']=copy_dataset['Marital_Status'].fillna("Married")


# In[43]:


#checking the result
copy_dataset.loc[301]


# In[44]:


# 7. checking for Home Owner Fetaure
nan_home=copy_dataset[copy_dataset["Home_Owner"].isnull()]
nan_home


# In[45]:


#replacing the nan value of Marital Status by calculating mode
mode_home=copy_dataset['Home_Owner'].mode()
copy_dataset['Home_Owner']=copy_dataset['Home_Owner'].fillna("Yes")


# In[46]:


#checking the result
copy_dataset.loc[365]


# In[47]:


# after replacing all those values 
clean_dataset=copy_dataset


# In[48]:


clean_dataset.shape


# In[49]:


# checking: If there are missing value in whole dataset or not
clean_dataset.isna().sum().sort_values(ascending=True)


# ## Transformation of Data

# In[50]:


# First of all spliting the categoric and numeric types
# 1. Categorical Columns
categoric_column=[x for x in clean_dataset.columns if clean_dataset[x].dtypes=="object"]
print(categoric_column)


# In[51]:


# 2. numeric columns
numeric_column=[y for y in clean_dataset.columns if clean_dataset[y].dtypes!="object"]
print(numeric_column)


# #### Converting Gender 'Yes'-1 and 'No'-0

# In[52]:


clean_dataset['Gender']=np.where(clean_dataset['Gender']=='Female',1,0)
clean_dataset['Gender'].dtypes


# In[53]:


clean_dataset['Gender']


# In[ ]:





# #### Converting Home Owner: Yes-1 and No-0

# In[54]:


clean_dataset['Home_Owner']=np.where(clean_dataset['Home_Owner']=='Yes',1,0)
clean_dataset['Home_Owner'].dtypes


# #### Converting Purchased Bike: Yes-1 and No-0

# In[55]:


clean_dataset['Purchased_Bike']=np.where(clean_dataset['Purchased_Bike']=="Yes",1,0)
clean_dataset['Purchased_Bike'].dtypes


# In[56]:


# Other Features Like Education, Occupation, Commute Distance, Region
# cannot be transform into 1,0 cause there is more selection 


# In[57]:


#if we look at the datatypes of dataset
display(clean_dataset.dtypes)


# In[58]:


# converting those float datatypes into int to understand easily
# the float datatypes are in : Children, Cars, Age
clean_dataset['Cars']=clean_dataset['Cars'].astype(int)


# In[59]:


clean_dataset['Cars'].dtypes


# In[60]:


# in the same converting fot those listed Colmuns too
clean_dataset['Children']=clean_dataset['Children'].astype(int)
clean_dataset['Age']=clean_dataset['Age'].astype(int)


# In[61]:


clean_dataset.dtypes


# # Feature Engineering

# In[62]:


#info of the dataset
clean_dataset.info()


# In[63]:


#checking the correlation between the columns in the dataset
#the age and children have highest relationship
correlation=clean_dataset.corr()
correlation


# In[64]:


#creating the covariance between different columns
covariance=clean_dataset.cov()
covariance


# In[65]:


# so using the heatmap we can to visualize the relationship between the featues
sns.heatmap(correlation,xticklabels=correlation.columns,yticklabels=correlation.columns,annot=True,
            cmap=sns.color_palette(as_cmap=True))


# In[66]:


# According to the heatmap the relationship between children and age is most and after that car and income.
# Since the correlation are lesser in other features


# In[67]:


# if we see in scatterplot between Children and Cars
sns.scatterplot(data=clean_dataset,x='Cars',y='Children',hue='Purchased_Bike')


# In[ ]:





# In[ ]:





# ## Data Visualization

# In[68]:


#importing the visualization tools
import matplotlib.pyplot as plt
#Histogarm is to visualize the distribution of numerical data
clean_dataset['Income'].plot(kind='hist')


# In[69]:


# another technique is to visualize the data using pie-chart
clean_dataset["Education"].value_counts().plot.pie()


# In[70]:


# historgram between the Purchased Bike and Income Feature
#sns.histplot(data=data_visual,x='Income',hue='Purchased Bike',bins=10,kde=True,color="m")


# In[71]:


# bar graph between Purchased Bike and Cars 
sns.barplot(data=clean_dataset,x='Purchased_Bike',y='Cars')


# In[72]:


# count plot show the counts of observations in each categorical bin using bars.
# displaying the counts as per the Cars Features
sns.countplot(data=clean_dataset,x='Cars', palette="dark")


# In[73]:


# showing the bar between Martial Status and Gender 
sns.countplot(data=clean_dataset,x="Marital_Status", hue="Purchased_Bike")


# In[74]:


sns.countplot(data=copy_dataset,x="Home_Owner", hue="Purchased_Bike")


# In[75]:


# visualize the data using pie-chart for Gender
clean_dataset["Gender"].value_counts().plot.pie()


# ## Filtering and Removing the Columns

# In[76]:


# The purchasing of bike as per the Person's Age
condition_young=clean_dataset[clean_dataset['Age']<28]
print("Number of Purchased_Bike or not for Young People\n",condition_young['Purchased_Bike'].value_counts())


# In[77]:


# in the same case but above 50
condition_old=clean_dataset[clean_dataset['Age']>50]
print("Number of Purchased_Bike or not by old people\n",condition_old['Purchased_Bike'].value_counts())


# In[78]:


# removing the ID from the dataset
clean_dataset.drop('ID',inplace=True,axis=1)


# In[79]:


clean_dataset


# In[ ]:





# # Using Classifier Algorithms
# 

# In[80]:


#creating the sample data
sample_data=clean_dataset.sample(50)
sample_data


# In[81]:


sample_data.shape


# In[82]:


#checking the nan value
sample_data.isnull().sum()


# In[83]:


# spliting the data
output=[]
for i in clean_dataset.columns:
    if clean_dataset[i].dtypes=='object':
        output.append(i)
        
# get.dummies convert categorial data into dummy or indicator variable
converted_data=pd.get_dummies(clean_dataset,columns=output)


# In[84]:


converted_data.tail()


# In[85]:


#importing the train_test_split library
from sklearn.model_selection import train_test_split

features=converted_data.drop("Purchased_Bike",axis=1)
targets=converted_data[['Purchased_Bike']]
x_train,x_test,y_train,y_test=train_test_split(features,targets,stratify=targets)


# In[2]:


x_train


# In[3]:


x_test


# In[4]:


y_train


# In[5]:


y_test


# ### 1. Logistic Regression

# In[90]:


#importing Logistic Regression 
from sklearn.linear_model import LogisticRegression
classifier_model=LogisticRegression(solver='sag',random_state=0)
classifier_model.fit(x_train,y_train)


# In[91]:


#creating prediction model
from sklearn.metrics import accuracy_score
prediction_model=classifier_model.predict(x_test)
accuracy=accuracy_score(y_test,prediction_model)
accuracy


# In[92]:


# if you change solver of Logistic regression to newton-cg
classifier_model=LogisticRegression(solver='newton-cg',random_state=0)
classifier_model.fit(x_train,y_train)


# In[93]:


# so if we test the accuracy then
prediction_model=classifier_model.predict(x_test)
accuracy=accuracy_score(y_test,prediction_model)
accuracy


# In[94]:


#creating the confusion matrix
confusion_matrix(y_test,prediction_model)


# In[95]:


print(classification_report(y_test,prediction_model))


# In[96]:


# then checking solver=lbfgs in the same model
classifier_model=LogisticRegression(solver='lbfgs',random_state=3)
classifier_model.fit(x_train,y_train)


# In[97]:


# so if we test the accuracy then
prediction_model=classifier_model.predict(x_test)
accuracy=accuracy_score(y_test,prediction_model)*100
accuracy


# In[98]:


#creating the confusion matrix
confusion_matrix(y_test,prediction_model)


# In[99]:


print(classification_report(y_test,prediction_model))


# ### 2. Support Vector Machine

# In[100]:


#importing the function
from sklearn.svm import SVC


# In[101]:


#fitting the model
svm_model=SVC()
svm_model.fit(x_train,y_train)


# In[102]:


# so if we test the accuracy then
prediction_model=svm_model.predict(x_test)
accuracy=accuracy_score(y_test,prediction_model)*100
accuracy


# In[103]:


#creating the confusion matrix
confusion_matrix(y_test,prediction_model)


# In[104]:


print(classification_report(y_test,prediction_model))


# ### 3. Random Forest 

# In[105]:


#importing the library
from sklearn.ensemble import RandomForestClassifier 


# In[106]:


#fitting the model
rf_model=RandomForestClassifier(n_estimators= 10, criterion="entropy")
rf_model.fit(x_train,y_train)


# In[107]:


# so if we test the accuracy then
prediction_model=rf_model.predict(x_test)
accuracy=accuracy_score(y_test,prediction_model)*100
accuracy


# In[108]:


#creating the confusion matrix
confusion_matrix(y_test,prediction_model)


# In[109]:


print(classification_report(y_test,prediction_model))


# # Comparision

# In[ ]:





# In[ ]:





# In[ ]:





# # Clustering

# In[110]:


#importing the important library
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


# In[111]:


plt.figure(1,figsize=(15,6))
n=0
for x in ['Age','Income','Purchased_Bike']:
    n+=1
    plt.subplot(1,3,n)
    plt.subplots_adjust(hspace=0.5,wspace=0.5)
    sns.distplot(converted_data[x],bins=15)
    plt.title('Distplot of {}'.format(x))
plt.show()


# In[112]:


sns.pairplot(converted_data,vars=['Age','Income','Cars'],hue="Purchased_Bike")


# In[113]:


#clustering based on Age and Income
plt.figure(1, figsize=(15,7))
plt.title("Scatter plot of Age vs Income",fontsize=20)
plt.xlabel('Age')
plt.ylabel('Income')
plt.scatter(x='Age',y='Income',data=converted_data,s=100)
plt.show()


# In[114]:


from sklearn.preprocessing import MinMaxScaler


# In[115]:


km=KMeans(n_clusters=3)
y_predicted=km.fit_predict(converted_data[['Age','Income']])
y_predicted


# In[116]:


converted_data['cluster']=y_predicted
converted_data.head()


# In[117]:


km.cluster_centers_


# In[118]:


df1=converted_data[converted_data.cluster==0]
df2=converted_data[converted_data.cluster==1]
df3=converted_data[converted_data.cluster==2]
plt.scatter(df1.Age,df1['Income'],color='yellow')
plt.scatter(df2.Age,df2['Income'],color='blue')
plt.scatter(df3.Age,df3['Income'],color='red')
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color="purple",marker="*",label="centroid")
plt.xlabel('Age')
plt.ylabel('Income')
plt.legend()


# In[119]:


scaler=MinMaxScaler()
scaler.fit(converted_data[['Income']])
converted_data['Income']=scaler.transform(converted_data[['Income']])

scaler.fit(converted_data[['Age']])
converted_data['Age']=scaler.transform(converted_data[['Age']])


# In[120]:


converted_data.head()


# In[121]:


plt.scatter(converted_data.Age,converted_data.Income)


# In[122]:


km=KMeans(n_clusters=3)
y_predicted=km.fit_predict(converted_data[['Age','Income']])
y_predicted


# In[123]:


converted_data['cluster']=y_predicted
converted_data.head()


# In[124]:


km.cluster_centers_


# In[125]:


df1=converted_data[converted_data.cluster==0]
df2=converted_data[converted_data.cluster==1]
df3=converted_data[converted_data.cluster==2]
plt.scatter(df1.Age,df1['Income'],color='yellow')
plt.scatter(df2.Age,df2['Income'],color='blue')
plt.scatter(df3.Age,df3['Income'],color='red')
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color="black",marker="*",label="centroid")
plt.xlabel('Age')
plt.ylabel('Income')
plt.legend()


# In[126]:


sse=[]
k_rng=range(1,10)
for k in k_rng:
    km=KMeans(n_clusters=k)
    km.fit(converted_data[['Age','Income']])
    sse.append(km.inertia_)


# In[127]:


plt.xlabel('K')
plt.ylabel('Sum of squiared error')
plt.plot(k_rng,sse)


# In[ ]:





# In[ ]:





# In[ ]:





# ## Importing the dataset into MongoDB
# 

# In[128]:


# step 1.1: ading the raw Carbon Nanotubes dataset
import csv
list=[]
with open("bike_buyers.csv","r") as file:
    data=csv.reader(file,delimiter='\n')
    #extracting one row
    for i in data:
        list.append(i[0].split(';'))#spliting the data wit delimiter
        


# In[129]:


# step 1.2: writing the updated data into a csv file
with open('new_updated_dataset.csv','w',newline='') as data:
    writer=csv.writer(data)
    writer.writerows(list)


# In[130]:


dataset.head()


# In[131]:


data=dataset.to_dict(orient='records')


# In[132]:


data


# In[133]:


# step 2.1: Establishing a connection with mongoDB on cloudbased server
import pymongo 
client=pymongo.MongoClient("mongodb+srv://admin:root@cluster0.vuiavbz.mongodb.net/?retryWrites=true&w=majority")
client.test


# In[134]:


#step 3 Creating the database
def checkExistence_DB(DB_NAME, client):
    """It verifies the existence of DB"""
    DBlist = client.list_database_names()
    if DB_NAME in DBlist:
        print(f"DB: '{DB_NAME}' exists")
        return True
    print(f"DB: '{DB_NAME}' not yet present present in the DB")
    return False


_ = checkExistence_DB(DB_NAME="bikePurchase", client=client)


# In[ ]:


# The database with above name is not present so we can create a database
db=client['bikePurchase']


# In[ ]:


print(db)


# In[ ]:


#creating the collection
COLLECTION_NAME='bigDataProject'
collection=db[COLLECTION_NAME]


# In[ ]:


#checking the collection already exist or not
def checkExistence_COL(COLLECTION_NAME, DB_NAME, db):
    """It verifies the existence of collection name in a database"""
    collection_list = db.list_collection_names()
    
    if COLLECTION_NAME in collection_list:
        print(f"Collection:'{COLLECTION_NAME}' in Database:'{DB_NAME}' exists")
        return True
    
    print(f"Collection:'{COLLECTION_NAME}' in Database:'{DB_NAME}' does not exists OR \n\
    no documents are present in the collection")
    return False


_ = checkExistence_COL(COLLECTION_NAME="bigDataProject", DB_NAME="bikePurchase", db=db)


# In[ ]:


#after checking lets insert documents into the collection
collection.insert_many(data)


# In[ ]:




