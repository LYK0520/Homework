from sklearn.metrics import confusion_matrix
from pandas import read_csv
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sn
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
import numpy as np
import imblearn
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
glass = pd.read_csv("../input/glass/glass.csv")
glass.head()
glass.shape
glass.info()
dups = glass.duplicated()
print('Number of duplicate rows: %d' % dups.sum())
print('Number of rows before discarding duplicates = %d' % glass.shape[0])
glass = glass.drop_duplicates()
print('Number of rows after discarding duplicates = %d' % glass.shape[0])
glass.describe()
glass['Type'].value_counts()
sn.set(style='whitegrid', font_scale=1.4)
plt.subplots(figsize=(12, 7))
sn.countplot(x='Type', data=glass, palette='Pastel1')
sn.boxplot(glass['RI'])
sn.boxplot(glass['Na'])
sn.boxplot(glass['Mg'])
sn.boxplot(glass['Al'])
sn.boxplot(glass['Si'])
sn.boxplot(glass['K'])
sn.boxplot(glass['Ca'])
sn.boxplot(glass['Ba'])
sn.boxplot(glass['Fe'])
sn.boxplot(glass['Type'])
sn.distplot(glass['RI'])
sn.distplot(glass['Na'])
sn.distplot(glass['Mg'])
sn.distplot(glass['Al'])
sn.distplot(glass['Si'])
sn.distplot(glass['K'])
sn.distplot(glass['Ca'])
sn.distplot(glass['Ba'])
sn.distplot(glass['Fe'])
sn.distplot(glass['Type'])
Y = 'Type'
X = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']
sn.heatmap(glass[X].isnull())
sn.heatmap(glass[X].corr())
glass[X].corr()
X = pd.DataFrame(glass.drop(["Type"], axis=1),
                 columns=['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe'])
Y = glass.Type
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=.2, random_state=30, stratify=Y)
n_neighbors = np.array(range(1, 40))
param_grid = dict(n_neighbors=n_neighbors)
param_grid
model = KNeighborsClassifier()
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=10)
grid.fit(X_train, Y_train)
print(grid.best_params_)
%matplotlib inline
# choose k between 1 to 41
k_range = range(1, 41)
k_scores = []
# use iteration to caclulator different k in models, then return the average accuracy based on the cross validation
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, Y_train, cv=10)
    k_scores.append(scores.mean())
# plot to see clearly
plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
plt.show()
model = KNeighborsClassifier(n_neighbors=1).fit(X_train, Y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(Y_test, y_pred)
print(accuracy)
confusion_matrix = confusion_matrix(Y_test, y_pred)
print(confusion_matrix)
print(classification_report(Y_test, y_pred))
sm = SMOTE(sampling_strategy='not majority', random_state=42)
x_resample, y_resample = sm.fit_resample(X, Y)
y_df = pd.DataFrame(y_resample)
y_df.value_counts()
X_train, X_test, Y_train, Y_test = train_test_split(
    x_resample, y_resample, test_size=.2, random_state=40, stratify=y_resample)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
n_neighbors = np.array(range(1, 40))
param_grid = dict(n_neighbors=n_neighbors)

model = KNeighborsClassifier()
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=10)
grid.fit(X_train, Y_train)
print(grid.best_params_)
%matplotlib inline
# choose k between 1 to 41
k_range = range(1, 41)
k_scores = []
# use iteration to caclulator different k in models, then return the average accuracy based on the cross validation
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, Y_train, cv=10)
    k_scores.append(scores.mean())
# plot to see clearly
plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
plt.show()
model = KNeighborsClassifier(n_neighbors=1).fit(X_train, Y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(Y_test, y_pred)
print(accuracy)
confusion_matrix = confusion_matrix(Y_test, y_pred)
confusion_matrix
print(classification_report(Y_test, y_pred))
