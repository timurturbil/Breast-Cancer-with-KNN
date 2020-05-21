import numpy as np
from sklearn.preprocessing import Imputer
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import cv2
import os
def loadImages(path = "."):
    
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.png')]
filenames = loadImages()
images = []
for file in filenames:
    images.append(cv2.imread(file, cv2.IMREAD_UNCHANGED))
images = pd.read_csv("Numerical_Datas.csv")
images = images.drop(['Image_Index'], axis=1)

y = np.array(images.Finding_Labels)
x = np.array(images.drop(['Finding_Labels'], axis=1))

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

a = np.array([[ 64,  60,  56, ...,  55,  59,  64],
       [ 65,  60,  57, ...,  54,  58,  62],
       [ 63,  65,  60, ...,  53,  57,  61],
       ...,
       [ 82,  83,  85, ..., 210, 215, 217],
       [ 86,  86,  82, ..., 211, 216, 218],
       [ 90,  89,  80, ..., 211, 215, 218]]).reshape(1,-1)
print(tahmin.predict(a))