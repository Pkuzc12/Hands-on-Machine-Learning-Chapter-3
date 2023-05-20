import os
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import SGDClassifier
from sklearn.base import BaseEstimator

DATA_PATH = os.path.join("datasets", "MNIST.csv")
mnist = pd.read_csv(DATA_PATH)

mnist_pixel = mnist.drop("class", axis=1, inplace=False)
mnist_label = mnist["class"].copy()
mnist_pixel_train = mnist_pixel.iloc[:60000]
mnist_pixel_test = mnist_pixel.iloc[60000:]
mnist_label_train = mnist_label.iloc[:60000].astype(np.uint8)
mnist_label_test = mnist_label.iloc[60000:].astype(np.uint8)
mnist_label_train_5 = (mnist_label_train == 5)
mnist_label_test_5 = (mnist_label_test == 5)

sgd_clf = SGDClassifier(random_state=42)
print(cross_val_score(sgd_clf, mnist_pixel_train, mnist_label_train_5, cv=3, scoring="accuracy"))


class Never5Classifier(BaseEstimator):
    def fit(self, X, y=None):
        pass

    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)


never_5_clf = Never5Classifier()
print(cross_val_score(never_5_clf, mnist_pixel_train, mnist_label_train_5, cv=3, scoring='accuracy'))
