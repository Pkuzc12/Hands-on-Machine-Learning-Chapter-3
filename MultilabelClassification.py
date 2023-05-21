import os
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier

DATA_PATH = os.path.join("datasets", "MNIST.csv")
mnist = pd.read_csv(DATA_PATH)

column = list(mnist)
column.remove("class")
mnist_train_pixel = mnist.iloc[:60000, :].loc[:, column]
mnist_train_class = mnist.iloc[:60000, :].loc[:, "class"].astype(np.uint8)
mnist_test_pixel = mnist.iloc[60000:, :].loc[:, column]
mnist_test_class = mnist.iloc[60000:, :].loc[:, "class"].astype(np.int8)
some_digit = mnist_train_pixel.iloc[0, :]

mnist_train_class_large = (mnist_train_class >= 7)
mnist_train_class_odd = (mnist_train_class % 2 == 1)
mnist_train_multilabel = np.c_[mnist_train_class_large, mnist_train_class_odd]

knn_clf = KNeighborsClassifier()
knn_clf.fit(mnist_train_pixel, mnist_train_multilabel)
print(knn_clf.predict([some_digit]))

mnist_train_knn_pred = cross_val_predict(knn_clf, mnist_train_pixel, mnist_train_multilabel, cv=3)
print(f1_score(mnist_train_multilabel, mnist_train_knn_pred, average="macro"))
