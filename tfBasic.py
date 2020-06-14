import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

# The fashion mnist dataset is included in Keras, thus just load it
fashion_mnist = keras.datasets.fashion_mnist

# Split the data into train/test
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


# Explore the dataset:
print(train_images.shape)
print(len(train_labels))
print(test_images.shape)
print(len(test_labels))

"""
class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


plt.figure()
plt.imshow(train_images[1])
plt.colorbar()
plt.grid(False)
plt.show()
"""

# Normalize the data set from 0-1
train_images = train_images / 255.0
test_images = test_images / 255.0

# Build an ANN structure, Define a 2 layers NN (i.e. 1 hidden layer only)
model = keras.Sequential(
    [
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(10, activation="softmax"),
    ]
)

"""
After creating the keras NN, we could get the summary to check this is what we want
print(model.summary())
"""

# Configures the model for traiing, the 'compile' method is passing hyperparameters to the NN
# which we could tune this parameters during training process
model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

# Fit the model with training data
# model.fit(train_images, train_labels, epochs=10, batch_size=None)
model.fit(train_images, train_labels, batch_size=None, epochs=1, verbose=1)

# Use the trained model and feed it with test data and see the accuracy on test data
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=1)
print("Test Accuracy: ", test_acc)
