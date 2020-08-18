# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 12:20:22 2020

@author: Anurag
"""
#Data Cleaning & Exploring
import numpy as np
import pandas as pd
data = pd.read_csv('CO2 Emissions_Canada.csv')
data1= pd.read_csv('CO2 Emissions_Canada.csv')
data.columns
data['Transmission'].describe()
data.describe()
data.isnull().any()
data.head()
data.hist(figsize=(12,8))
d1=  data.iloc[:,11]

#Renaming the columns in the data set 
data.rename(columns={
                   'Vehicle Class': 'Vehicle_Class'},
          inplace=True, errors='raise')

#Using count encoding on categorical data
data_frequency_map2 = data.Fuel_type.value_counts().to_dict()
data.Fuel_type = data.Fuel_type.map(data_frequency_map2)
data.head()

#Transforming the categorical values of the data into dummies
df2 = pd.DataFrame(pd.get_dummies(data1))
y1 = pd.DataFrame(df2, columns = ['CO2 Emissions(g/km)']) 

#Creating new data frame 
df1 = pd.DataFrame(data, columns = ['Model', 'Make', 'Vehicle Class'])

# Finding the outliers using boxplots
data.boxplot(figsize=(22,15))


#Function to find the outliers in the data set
outliers= []
def find_outliers(dat):
    threshold = 3
    mean = np.mean(dat)
    std = np.std(dat)
        
    for i in dat:
        z_score = (i-mean)/std
            
        if np.abs(z_score) > threshold:
            outliers.append(i)
    return outliers
    
put_pt = find_outliers(d1)    
print(put_pt)        
        
#Finding outliers using scatter plots
import matplotlib.pyplot as plt
plt.scatter(data['Fuel Consumption Comb (mpg)'], data['CO2 Emissions(g/km)'])

#Test to find the influence of different variables on co2 emissions
w = data.corr()


#Univariate feature selection method 

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

X1 = data.iloc[:, 0:11 ]#Independent columns name
y1 = data.iloc[:, 11] # Target columns name

bestfeatures = SelectKBest(score_func=chi2, k=10)
fit = bestfeatures.fit(X1,y1)

dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X1.columns)
#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
featureScores
print(featureScores.nlargest(10,'Score')) 

# Feature Importance method

from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
model = ExtraTreesClassifier()
model.fit(X1,y1)
print(model.feature_importances_) #

feat_importances = pd.Series(model.feature_importances_, index=X1.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()


#Correlation Matrix with Heatmap

import seaborn as sns
#get correlations of each features in dataset
corrmat = data.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(data[top_corr_features].corr(),annot=True,cmap="RdYlGn")

# The best feature of the data using the corelation matrix is 
# 1-  fuel consumption city
# 2-  fuel consumption combined
# 3-  fuel consumption highway
# 4-  Engine size
# 5-  Cylinders 
# 6-  Make
# 7-  Transmission
# 8-  Fuel type
# 9-  Veichle class
# 10- Model


'''
Creating a model using simple linear regression to evaluate the
difference on co2 emission when fuel consumption is considered
'''





x_1 = pd.DataFrame(data, columns = ['Fuel Consumption City (L/100 km)','Fuel Consumption Hwy (L/100 km)'])

x_2 = pd.DataFrame(data, columns=['Fuel Consumption Comb (L/100 km)'])




#Creating the Model using the x_1 as independent variable


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train1, X_test1, y_train1, y_test1 = train_test_split(x_1, y1, test_size = 0.25, random_state = 42)



from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train1, y_train1)



y_pred1 = regressor.predict(X_test1)


from sklearn.metrics import explained_variance_score
explained_variance_score(y_test1, y_pred1)



#Creating the model using the x_1 as independent variable


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train2, X_test2, y_train2, y_test2 = train_test_split(x_2, y1, test_size = 0.25, random_state = 42)



from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train2, y_train2)



y_pred2 = regressor.predict(X_test2)


from sklearn.metrics import explained_variance_score
explained_variance_score(y_test2, y_pred2)

# This explained variance  test concludes that the x_1 data frame performs slightly better than
# x_2
# x_1 0.84768
# x-2 0.83822

# save the model to disk
import pickle
filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))



  