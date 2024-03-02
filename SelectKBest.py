import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import load_breast_cancer
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
sel = SelectKBest(f_classif, k=4) #use K-best function
X_train_t = sel.fit(X_train, y_train) #apply the function
X_test_t = sel.transform(X_test)


print("All features:")
print(x.columns.values)

x_train_2 = sel.transform(X_train)
x_test_2 = sel.transform(X_test)
#random forest classifier with n_estimators=10 (default)
clf_rf_2 = RandomForestClassifier()
clr_rf_2 = clf_rf_2.fit(x_train_2,y_train)
ac_2 = accuracy_score(y_test,clf_rf_2.predict(x_test_2))
print('Accuracy is: ',ac_2)

scores = -np.log10(sel.pvalues_)
scores /= scores.max()
X_indices = np.arange(x.shape[-1])
plt.figure(1)
plt.clf()
plt.bar(X_indices - 0.05, scores, width=0.2)
plt.title("Feature univariate score")
plt.xlabel("Feature number")
plt.ylabel(r"Univariate score ($-Log(p_{value})$)")
plt.show()

#pd.Series(sel.feature_performance_).sort_values(ascending=False).plot.bar(figsize=(10, 5))
'''
plt.ylabel('Performance')
plt.title('Univariate performance')
plt.xticks(rotation=45,horizontalalignment='right',fontweight='light')
plt.show()
'''
