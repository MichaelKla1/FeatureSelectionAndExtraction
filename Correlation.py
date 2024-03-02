import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from feature_engine.selection import SelectBySingleFeaturePerformance
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap

# load dataset
ds = pd.read_csv("Applicant-details.csv")
yes_no_map = {'yes': 1, 'no': 0}
status_map = {'married': 1, 'single': 0}
house_rent_map = {'owned': 3, 'rented': 1, 'norent_noown': 2}
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

y = ds['Loan_Default_Risk']  # output variable
x = ds.drop(['Loan_Default_Risk', 'Residence_City', 'Applicant_ID'],axis=1)  # features to drop
X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0)  # set training and test data
threshold = 0.5  # Define threshold
corr_matrix = x.corr().abs()  # Absolute value correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
X_train = X_train.drop(columns=to_drop)  # drop columns
X_test = X_test.drop(columns=to_drop)  # drop columns
sc = StandardScaler()
X_train = sc.fit_transform(X_train)  # standartize the data
X_test = sc.transform(X_test)  # standartize the data
classifier = RandomForestClassifier()
classifier.fit(X_train, y_train)  # train model with 4 principal components
score = classifier.score(X_test, y_test)  # get accuracy
print("Accuracy: ", score)

f, ax = plt.subplots(figsize=(18, 18))
sns.heatmap(x.corr(), annot=True, linewidths=.5, fmt='.1f', ax=ax)  # build correlation heatmap
plt.show()  # show heatmam

exit()

pca = PCA(n_components=4)  # apply pca with 4 principal components
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
classifier = RandomForestClassifier()
classifier.fit(X_train, y_train)  # train model with 4 principal components
score = classifier.score(X_test, y_test)  # get accuracy
print("Accuracy: ", score)

exit()
