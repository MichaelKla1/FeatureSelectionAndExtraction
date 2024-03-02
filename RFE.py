import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from feature_engine.selection import SelectBySingleFeaturePerformance
# load dataset
ds = pd.read_csv("Applicant-details.csv")
yes_no_map = {'yes': 1, 'no': 0}
status_map = {'married': 1, 'single': 0}
house_rent_map = {'owned':3,'rented':1,'norent_noown':2}
ds['Vehicle_Ownership(car)'] = ds['Vehicle_Ownership(car)'].map(yes_no_map)
ds['House_Ownership'] = ds['House_Ownership'].map(house_rent_map)
ds['Marital_Status'] = ds['Marital_Status'].map(status_map)
ds = pd.get_dummies(ds, columns = ['Occupation', 'Residence_State'])
'''
map_arr = []
for item in ds['Occupation']:
    if not item in map_arr:
        map_arr.append(item)
occupation_map = {}
i=0
while i<len(map_arr):
    occupation_map[map_arr[i]]=i
    i+=1
ds['Occupation'] = ds['Occupation'].map(occupation_map)

map_arr = []
for item in ds['Residence_State']:
    if not item in map_arr:
        map_arr.append(item)
residence_state_map = {}
i=0
while i<len(map_arr):
    residence_state_map[map_arr[i]]=i
    i+=1
ds['Residence_State'] = ds['Residence_State'].map(residence_state_map)
'''

y = ds['Loan_Default_Risk'] #output variable
x = ds.drop(['Loan_Default_Risk','Residence_City','Applicant_ID'],axis=1) #features to drop
X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0) #set training and test data
model = RandomForestClassifier(n_estimators=100, random_state=42) #base model
rfe = RFE(estimator=model, n_features_to_select=4) #initialize rfe
rfe.fit(X_train, y_train) #fit rfe

# Get the ranking of each feature
feature_ranking = rfe.ranking_

# Get the selected features
selected_features = np.where(feature_ranking == 1)[0]

print("All features:")
print(x.columns.values)

# Print the selected features
print("Selected Features:", selected_features)

# Train the final model using the selected features
i=0
j=0
while i < len(X_train.columns):
    if j not in selected_features:
        X_train=X_train.drop(X_train.columns[i],axis=1)
        X_test = X_test.drop(X_test.columns[i], axis=1)
        i-=1
    i+=1
    j+=1
model.fit(X_train, y_train)

# Evaluate the model on the test set
accuracy = model.score(X_test, y_test)
print("Accuracy on the Test Set:", accuracy)

# Visualize the feature ranking
plt.figure(figsize=(10, 6))
plt.title("RFE - Feature Ranking")
plt.xlabel("Feature Index")
plt.ylabel("Ranking")
plt.bar(range(len(feature_ranking)), feature_ranking)
plt.show()



