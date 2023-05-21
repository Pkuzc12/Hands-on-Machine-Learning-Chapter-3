import os
import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt

DATA_PATH = os.path.join("datasets", "MNIST.csv")
mnist = pd.read_csv(DATA_PATH)

column = list(mnist)
column.remove("class")
mnist_train_pixel = mnist.iloc[60000:, :].loc[:, column]
mnist_train_class = mnist.iloc[60000:, :].loc[:, "class"]
mnist_test_pixel = mnist.iloc[60000:, :].loc[:, column].astype(np.uint8)
mnist_test_class = mnist.iloc[60000:, :].loc[:, "class"].astype(np.uint8)

sgd_clf = SGDClassifier(random_state=42)

mnist_train_predict = cross_val_predict(sgd_clf, mnist_train_pixel, mnist_train_class, cv=3)
conf_mx = confusion_matrix(mnist_train_class, mnist_train_predict)
print(conf_mx)

plt.matshow(conf_mx, cmap=plt.cm.gray)
plt.show()

row_sums = conf_mx.sum(axis=1, keepdims=True)
norm_conf_mx = conf_mx / row_sums

np.fill_diagonal(norm_conf_mx, 0)
plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
plt.show()
