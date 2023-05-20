import os
import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

DATA_PATH = os.path.join("datasets", "MNIST.csv")

mnist = pd.read_csv(DATA_PATH)
mnist_train = mnist.iloc[:60000]
mnist_test = mnist.iloc[60000:]
columns = list(mnist)
columns.remove('class')
mnist_train_pixel = mnist_train.loc[:, columns]
mnist_train_target = mnist_train['class'].astype(np.uint8)
mnist_test_pixel = mnist_test.loc[:, columns]
mnist_test_target = mnist_test['class'].astype(np.uint8)
mnist_train_target_5 = (mnist_train_target == 5)
mnist_test_target_5 = (mnist_test_target == 5)

sgd_clf = SGDClassifier(random_state=42)
mnist_train_predict = cross_val_predict(sgd_clf, mnist_train_pixel, mnist_train_target_5, cv=3)
print(confusion_matrix(mnist_train_target_5, mnist_train_predict))
print(precision_score(mnist_train_target_5, mnist_train_predict))
print(recall_score(mnist_train_target_5, mnist_train_predict))
print(f1_score(mnist_train_target_5, mnist_train_predict))
