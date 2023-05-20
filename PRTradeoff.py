import os
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, precision_score, recall_score
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_predict
from matplotlib import pyplot as plt

DATA_PATH = os.path.join("datasets", "MNIST.csv")
mnist = pd.read_csv(DATA_PATH)

columns = list(mnist)
columns.remove("class")
mnist_train_pixel = mnist.loc[:, columns].iloc[:60000, :]
mnist_train_class = mnist["class"].iloc[:60000]
mnist_train_class_5 = (mnist_train_class.astype(np.uint8) == 5)
mnist_test_pixel = mnist.loc[:, columns].iloc[60000:, :]
mnist_test_class = mnist["class"].iloc[60000:]
mnist_test_class_5 = (mnist_test_class.astype(np.uint8) == 5)

sgd_clf = SGDClassifier(random_state=42)
decision_scores = cross_val_predict(sgd_clf, mnist_train_pixel, mnist_train_class_5, cv=3, method="decision_function")
precision, recalls, thresholds = precision_recall_curve(mnist_train_class_5, decision_scores)


# def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
#     plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
#     plt.plot(thresholds, recalls[:-1], "g-", label="Recall")


# plot_precision_recall_vs_threshold(precision, recalls, thresholds)
# plt.show()

threshold_90_precision = thresholds[np.argmax(precision >= 0.90)]
mnist_train_pred_90 = (decision_scores >= threshold_90_precision)
print(precision_score(mnist_train_class_5, mnist_train_pred_90))
print(recall_score(mnist_train_class_5, mnist_train_pred_90))
