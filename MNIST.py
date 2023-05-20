import os
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', version=1, parser='auto')
# print(mnist.keys())
# print(mnist["data"])
# print(mnist["target"])
# print(mnist["frame"])
# print(mnist["categories"])
# print(mnist["feature_names"])
# print(mnist["target_names"])
# print(mnist["DESCR"])
# print(mnist["details"])
# print(mnist["url"])
# print(type(mnist["data"]))
# print(type(mnist["target"]))
# print(type(mnist["frame"]))
some_digit = mnist["data"].values[0]
some_digit_image = some_digit.reshape(28, 28)

plt.imshow(some_digit_image, cmap='binary')
plt.show()
print(mnist["target"].values[0])

mnist["frame"].to_csv(os.path.join("datasets", "MNIST.csv"), index=False)
