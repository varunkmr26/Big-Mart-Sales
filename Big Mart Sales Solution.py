import sys

import numpy as np

import pandas as pd

import scipy as sp

import matplotlib.pyplot as plt

import matplotlib.pylab as pylab

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')

from sklearn import model_selection

from sklearn import feature_selection

from sklearn import metrics

from sklearn import linear_model, neighbors, svm, tree, ensemble

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

train_data = pd.read_csv('train.csv')

test_data = pd.read_csv('test.csv')

full_data = [train_data, test_data]

train_data.info()

test_data.info()

print(train_data.head(10))

print(test_data.head(10))

print(train_data['Outlet_Size'].notnull())

print(train_data['Outlet_Size'].value_counts())

plt.figure(figsize = [10,10])

plt.subplot(221)

plt.hist(train_data['Item_Weight'].notnull())
plt.title('Item Weight Nulls')

plt.subplot(222)

plt.hist(train_data[train_data['Outlet_Size']=='Small']['Outlet_Location_Type'])
plt.title('Small vs Location Type')

plt.subplot(223)

plt.hist(train_data[train_data['Outlet_Size']=='Medium']['Outlet_Location_Type'])
plt.title('Medium vs Location Type')

plt.subplot(224)

plt.hist(train_data[train_data['Outlet_Size']=='High']['Outlet_Location_Type'])
plt.title('High vs Location Type')

fig, ax = plt.subplots(figsize=[8,6])

sns.barplot(train_data['Item_Type'],
            train_data[train_data['Item_Weight'].notnull()==True]['Item_Weight'], ax=ax)

fig, axs = plt.subplots(figsize=[8,6])

sns.barplot(train_data['Item_Fat_Content'],
            train_data[train_data['Item_Weight'].notnull()==True]['Item_Weight'], ax=axs)

fig, saxis = plt.subplots(figsize=[16,12])

matrix = train_data.corr()

colormap = sns.diverging_palette(220,10,as_cmap = True)

sns.heatmap(matrix, vmax = 0.8, cmap = colormap, square = True, annot = True, ax = saxis)


target = train_data['Item_Outlet_Sales']

train_data.drop('Item_Outlet_Sales',axis=1,inplace=True)

print(train_data[train_data['Outlet_Location_Type'] == 'Tier 2']['Outlet_Size'].isnull().sum())

test_id = test_data['Item_Identifier']

test_outlet_id = test_data['Outlet_Identifier']

for data in full_data:
    data.drop(['Outlet_Establishment_Year','Item_Identifier'],axis=1, inplace=True)
    data['Item_Weight'].fillna(data['Item_Weight'].median(),inplace = True)
    data.loc[(data['Outlet_Size'].isnull()) & (data['Outlet_Location_Type'] == 'Tier 2'), 'Outlet_Size']='Small'
    data.loc[(data['Outlet_Size'].isnull()) & (data['Outlet_Location_Type'] == 'Tier 3'), 'Outlet_Size']='Medium'
    #data['Item_Weight'] = pd.qcut(data['Item_Weight'], 4)
    #data['Item_MRP'] = pd.qcut(data['Item_MRP'],4)
    
plt.figure(figsize = [10,10])

plt.subplot(211)

plt.boxplot(train_data['Item_Weight'], showmeans=True, meanline= True)
plt.title('Item Weight Distribution')

plt.subplot(212)

plt.boxplot(train_data['Item_MRP'], showmeans=True, meanline= True)
plt.title('Item MRP Distribution')

train_data.info()

test_data.info()
    
label = LabelEncoder()

std = StandardScaler()

print(train_data['Outlet_Size'].value_counts())

object_list = list(train_data.select_dtypes(include = ['object']).columns)

dummies = pd.get_dummies(train_data[object_list], prefix = object_list)

#train_data.drop(object_list, axis = 1, inplace = True)

#train_data = pd.concat([train_data,dummies], axis =1)

#test_data.drop(object_list, axis = 1, inplace = True)

#test_data = pd.concat([train_data,dummies], axis =1)

for data in full_data:
    data['Item_Fat_Content'] = label.fit_transform(data['Item_Fat_Content'])
    data['Item_Type'] = label.fit_transform(data['Item_Type'])
    #data['Item_Weight'] = label.fit_transform(data['Item_Weight'])
    #data['Item_MRP'] = label.fit_transform(data['Item_MRP'])
    data['Outlet_Size'] = label.fit_transform(data['Outlet_Size'])
    data['Outlet_Location_Type'] = label.fit_transform(data['Outlet_Location_Type'])
    data['Outlet_Type'] = label.fit_transform(data['Outlet_Type'])
    #data['Item_Identifier'] = label.fit_transform(data['Item_Identifier'])
    data['Outlet_Identifier'] = label.fit_transform(data['Outlet_Identifier'])
    
fig, axis = plt.subplots(figsize = [8,6])

sns.distplot(train_data['Item_Visibility'],ax = axis)
    
print(train_data.head(10))

cv = model_selection.ShuffleSplit(n_splits=10,test_size = 0.2, train_size = 0.8, random_state = 0)

x_train, x_cv, y_train, y_cv = model_selection.train_test_split(train_data,target,test_size = 0.2, random_state = 0)

MLA = [linear_model.LinearRegression(),
       linear_model.Lasso(),
       linear_model.Ridge(),
       tree.DecisionTreeRegressor(),
       ensemble.RandomForestRegressor(n_estimators = 100, oob_score = True, random_state = 0),
       ensemble.GradientBoostingRegressor(n_estimators = 100,random_state = 0)]
       #svm.SVR()]

for alg in MLA:
    cv_results = model_selection.cross_validate(alg,x_train,y_train,scoring = 'neg_mean_squared_error',cv = cv)
    print(alg.__class__.__name__)
    print("\nTrain Score: ", cv_results['train_score'].mean())
    print("\ntest Score: ", cv_results['test_score'].mean())
    print("\nTest Score STD: ", cv_results['test_score'].std())
    alg.fit(x_train,y_train)
    predict = alg.predict(x_cv)
    print("\nCV Score: ", np.sqrt(metrics.mean_squared_error(y_cv,predict)))
    
gbm = ensemble.GradientBoostingRegressor(n_estimators=100,random_state=0)

gbm.fit(x_train,y_train)

test_data['Item_Outlet_Sales'] = gbm.predict(test_data)

submission = pd.DataFrame(data={'Item_Identifier':test_id,'Outlet_Identifier':test_outlet_id,'Item_Outlet_Sales':test_data['Item_Outlet_Sales']})

submission.to_csv('BMS.csv')
    
