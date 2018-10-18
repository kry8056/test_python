from sklearn import datasets

X, y = datasets.make_classification(n_samples=100, n_features=2,
                                    n_redundant=0, n_classes=2,
                                    random_state=7816)

import matplotlib.pyplot as plt
#%matplotlib
plt.scatter(X[:, 0], X[:, 1], c=y, s=100)
plt.xlabel('x values')
plt.ylabel('y values')

import numpy as np
X = X.astype(np.float32)
y = y * 2 - 1

from sklearn import model_selection as ms
X_train, X_test, y_train, y_test = ms.train_test_split(
        X, y, test_size=0.2, random_state=42
        )

import cv2
svm = cv2.ml.SVM_create()
#partition the data with a straight line
svm.setKernel(cv2.ml.SVM_LINEAR)
#train SVM
svm.train(X_train, cv2.ml.ROW_SAMPLE, y_train)
_, y_pred = svm.predict(X_test)
from sklearn import metrics
score=metrics.accuracy_score(y_test, y_pred)

