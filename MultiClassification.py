import os
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

DATA_PATH = os.path.join("datasets", "MNIST.csv")
mnist = pd.read_csv(DATA_PATH)

column = list(mnist)
column.remove("class")
mnist_train_pixel = mnist.iloc[:60000, :].loc[:, column]
mnist_test_pixel = mnist.iloc[60000:, :].loc[:, column]
mnist_train_class = mnist.iloc[:60000, :].loc[:, "class"].astype(np.uint8)
mnist_test_class = mnist.iloc[60000:, :].loc[:, "class"].astype(np.uint8)

some_digit = mnist_train_pixel.iloc[0, :]

svm_clf = SVC()
svm_clf.fit(mnist_train_pixel, mnist_train_class)
print(svm_clf.predict([some_digit]))
print(svm_clf.decision_function([some_digit]))
print(svm_clf.classes_)

ovr_svm_clf = OneVsRestClassifier(SVC())
ovr_svm_clf.fit(mnist_train_pixel, mnist_train_class)
print(ovr_svm_clf.predict([some_digit]))
print(len(ovr_svm_clf.estimators_))

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(mnist_train_pixel, mnist_train_class)
print(sgd_clf.predict([some_digit]))
print(sgd_clf.decision_function([some_digit]))
print(cross_val_score(sgd_clf, mnist_train_pixel, mnist_train_class, cv=3, scoring="accuracy"))

scalar = StandardScaler()
mnist_train_pixel_scaled = scalar.fit_transform(mnist.iloc[:60000, :].loc[:, column].astype(np.float64))
print(cross_val_score(sgd_clf, mnist_train_pixel_scaled, mnist_train_class, cv=3, scoring="accuracy"))
