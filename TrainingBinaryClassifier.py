import os
import pandas as pd
from sklearn.linear_model import SGDClassifier
import numpy as np

DATA_PATH = "datasets"


def load_data(data_path=DATA_PATH):
    mnist_path = os.path.join(data_path, "MNIST.csv")
    return pd.read_csv(mnist_path)


mnist = load_data()
mnist_pixel = mnist.drop("class", axis=1, inplace=False).values
mnist_label = mnist["class"].values
mnist_train_pixel = mnist_pixel[:60000]
mnist_test_pixel = mnist_pixel[60000:]
mnist_train_label = mnist_label[:60000]
mnist_test_label = mnist_label[60000:]
mnist_train_label_5 = (mnist_train_label.astype(np.uint8) == 5)
mnist_test_label_5 = (mnist_test_label.astype(np.uint8) == 5)
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(mnist_train_pixel, mnist_train_label_5)

print(sgd_clf.predict([mnist_train_pixel[0]]))
