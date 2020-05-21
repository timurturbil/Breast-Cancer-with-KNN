import numpy as np
from sklearn.preprocessing import Imputer
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

veri = pd.read_csv("breast-cancer.csv")
veri = veri.drop(['id'], axis=1)

y = np.array(veri.diagnosis)
x = np.array(veri.drop(['diagnosis'], axis=1))

imp = Imputer(missing_values=-99999, strategy="mean", axis=0)
x = imp.fit_transform(x)
"""
for z in range(25):
z = 2*z+1
print(z, " hesaplama")
tahmin = KNeighborsClassifier(n_neighbors=3, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='euclidean', metric_params=None, n_jobs=1)
tahmin.fit(x,y)
ytahmin = tahmin.predict(x)

basari = accuracy_score(y, ytahmin, normalize=True, sample_weight=None)
print(basari)
"""

X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.2)

tahmin = KNeighborsClassifier(n_neighbors=3, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='euclidean', metric_params=None, n_jobs=1)
tahmin.fit(X_train,y_train)
basari= tahmin.score(X_test, y_test)
print("yüzde", basari*100," oranında:")

a = np.array([9.509,12.44,60.34,273.9,0.1025,0.06492,0.02956,0.02076,0.1815,0.06905,0.2773,0.9768,1.909,15.7,0.009606,0.01432,0.01985,0.01421,0.02027,0.002968,10.23,15.66,65.13,314.9,0.1324,0.1148,0.08867,0.06227,0.245,0.07773]).reshape(1,-1)
print(tahmin.predict(a))