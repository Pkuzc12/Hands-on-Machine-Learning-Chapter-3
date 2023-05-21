import os
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot as plt

DATA_PATH = os.path.join("datasets", "MNIST.csv")
mnist = pd.read_csv(DATA_PATH)

column = list(mnist)
column.remove("class")
mnist_train_pixel = mnist.iloc[60000:, :].loc[:, column]
mnist_test_pixel = mnist.iloc[:60000, :].loc[:, column]

noise = np.random.randint(0, 100, (len(mnist_train_pixel), 784))
mnist_train_pixel_mod = mnist_train_pixel+noise
noise = np.random.randint(0, 100, (len(mnist_test_pixel), 784))
mnist_test_pixel_mod = mnist_test_pixel+noise

plt.imshow(mnist_train_pixel_mod.iloc[3, :].values.reshape(28, 28), cmap=plt.cm.gray)
plt.show()

knn_clf = KNeighborsClassifier()
knn_clf.fit(mnist_train_pixel_mod, mnist_train_pixel)
digit_clean = knn_clf.predict([mnist_train_pixel_mod.iloc[3, :]])
plt.imshow(digit_clean.reshape(28, 28), cmap=plt.cm.gray)
plt.show()
