import math as m
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as datasets
from tensorflow import keras, math, random, stack
from tensorflow.keras import layers, initializers
from tensorflow.keras.activations import relu


# 
#       x[0] --- h1 
#            \ /    \
#             X       output
#            / \    /
#       x[1] --- h2
#
# This is the base model - nothing fancy here

# modify this line
ACTIVATION = "sigmoid"

# DONT MODIFY
# this is just for declaring
LABEL = None
THRESH = None
# Set random seed for reproducibility
np.random.seed(1)
random.set_seed(1)

if ACTIVATION=="tanh":
    LABEL = -1
    THRESH = 0
if ACTIVATION=="sigmoid":
    LABEL = 0
    THRESH = 0.5


def custom_activation(x):
    return x**2


model = keras.models.Sequential()
model.add(layers.Dense(1, input_dim=1, activation=custom_activation))
model.compile(loss="mean_squared_error")

log = np.array([1, 4])
X = -10.0 + 20.0 * np.random.random(100)
Y = log[0] + log[1] * X**2 + np.random.randn(100)

plt.scatter(X, Y, s=100, alpha=.9)
plt.show()

history = model.fit(X, Y, batch_size=50, epochs=1000)

meshData = np.linspace(-10,10,50)

fig, ax = plt.subplots()
Z = model.predict(meshData)
plt.plot(meshData, Z)
ax.scatter(X, Y, s=100, alpha=.9)
plt.show()
