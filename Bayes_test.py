#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 08:03:08 2018

@author: ykarpitski
"""

from sklearn import datasets
X, y = datasets.make_blobs(100, 2, centers=2,
random_state=1701, cluster_std=2)

import matplotlib.pyplot as plt
plt.style.use('ggplot')
#%matplotlib inline
plt.scatter(X[:, 0], X[:, 1], c=y, s=50);


import numpy as np
from sklearn import model_selection as ms
X = X.astype(np.float32)
X_train, X_test, y_train, y_test = ms.train_test_split(
        X, y, test_size=0.1)

import cv2
model_norm = cv2.ml.NormalBayesClassifier_create()

model_norm.train(X_train, cv2.ml.ROW_SAMPLE, y_train)

_, y_pred = model_norm.predict(X_test)

from sklearn import metrics
score=metrics.accuracy_score(y_test, y_pred)



