import os
import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_curve, roc_auc_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot as plt

DATA_PATH = os.path.join("datasets", "MNIST.csv")
mnist = pd.read_csv(DATA_PATH)

columns = list(mnist)
columns.remove("class")
mnist_train_pixel = mnist.iloc[:60000, :].loc[:, columns]
mnist_train_class = (mnist.iloc[:60000, :].loc[:, "class"].astype(np.uint8) == 5)
mnist_test_pixel = mnist.iloc[60000:, :].loc[:, columns]
mnist_test_class = (mnist.iloc[60000:, :].loc[:, "class"].astype(np.uint8) == 5)

sgd_clf = SGDClassifier(random_state=42)
mnist_train_scores = cross_val_predict(sgd_clf, mnist_train_pixel, mnist_train_class, cv=3, method="decision_function")
fpr, tpr, thresholds = roc_curve(mnist_train_class, mnist_train_scores)


def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis("on")
    plt.grid("on")


# plot_roc_curve(fpr, tpr)
# plt.show()

forest_clf = RandomForestClassifier(random_state=42)
mnist_train_prob = cross_val_predict(forest_clf, mnist_train_pixel, mnist_train_class, cv=3, method="predict_proba")
mnist_train_scores_forest = mnist_train_prob[:, 1]
fpr_forest, tpr_forest, thresholds_forest = roc_curve(mnist_train_class, mnist_train_scores_forest)

plt.plot(fpr, tpr, "b:", label="SGD")
plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")
plt.legend(loc="lower right")
plt.show()

mnist_train_predict_forest = cross_val_predict(forest_clf, mnist_train_pixel, mnist_train_class, cv=3)

print(roc_auc_score(mnist_train_class, mnist_train_scores))
print(roc_auc_score(mnist_train_class, mnist_train_scores_forest))
print(precision_score(mnist_train_class, mnist_train_predict_forest))
print(recall_score(mnist_train_class, mnist_train_predict_forest))
