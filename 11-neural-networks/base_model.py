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
    return relu(x, threshold=0)


model = keras.models.Sequential()
model.add(layers.Dense(2, input_dim=2, activation=custom_activation))
model.add(layers.Dense(1, activation=ACTIVATION))
#opt = keras.optimizers.Adam(learning_rate=.01)
model.compile(loss="binary_crossentropy") #, optimizer=opt)

centers = [[0, 0]]
t, _ = datasets.make_blobs(n_samples=200, centers=centers, cluster_std=1,
                                random_state=1)

# CIRCLE
def generate_circle_data(t):
    # create some space between the classes
    X = np.array(list(filter(lambda x : (x[0] - centers[0][0])**2 + (x[1] - centers[0][1])**2 < 1 or (x[0] - centers[0][0])**2 + (x[1] - centers[0][1])**2 > 1.5, t)))
    Y = np.array([1 if (x[0] - centers[0][0])**2 + (x[1] - centers[0][1])**2 >= 1 else LABEL for x in X])
    return X, Y

# LINE
def generate_line_data(t):
    # create some space between the classes
    X = np.array(list(filter(lambda x : x[0] - x[1] < -.5 or x[0] - x[1] > .5, t)))
    Y = np.array([1 if x[0] - x[1] >= 0 else LABEL for x in X])
    return X, Y

# CURVE
def generate_curve_data(t):
    # create some space between the classes
    X = np.array(list(filter(lambda x : m.cos(4*x[0]) - x[1] < -.5 or m.cos(4*x[0]) - x[1] > .5, t)))
    Y = np.array([1 if m.cos(4*x[0]) - x[1] >= 0 else LABEL for x in X])
    return X, Y

# XOR
def generate_xor_data():
    X = np.array([
        [0,0],
        [0,1],
        [1,0],
        [1,1]])
    Y = np.array([x[0]^x[1] for x in X])
    return X, Y

X, Y = generate_circle_data(t)
# X, Y = generate_line_data(t)
# X, Y = generate_curve_data(t)
# X, Y = generate_xor_data()

colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
colors = np.hstack([colors] * 20)

plt.scatter(X[:,0],X[:,1],color=colors[Y].tolist(), s=100, alpha=.9)
plt.show()

history = model.fit(X, Y, batch_size=50, epochs=1000)

# Show the transformation of the input at the first hidden layer
layer = model.layers[0]
print(layer.get_config(), layer.get_weights())
keras_function = keras.backend.function([model.input], [layer.output])
layerVals = np.array(keras_function(X))[0]
plt.scatter(layerVals[:,0], layerVals[:, 1], color=colors[Y].tolist(), s=100, alpha=.9)
plt.show()

# create a mesh to plot in
h = .02  # step size in the mesh
x_min, x_max = layerVals[:, 0].min() - .5, layerVals[:, 0].max() + 1
y_min, y_max = layerVals[:, 1].min() - .5, layerVals[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
meshData = np.c_[xx.ravel(), yy.ravel()]

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh
fig, ax = plt.subplots()
layer = model.layers[-1]

intermediateModel = keras.models.Sequential()
intermediateModel.add(layers.Dense(1, input_dim=2, activation=ACTIVATION))
intermediateModel.compile(loss="binary_crossentropy")
intermediateModel.layers[0].set_weights(layer.get_weights())

Z = intermediateModel.predict(meshData)
Z = np.array([LABEL if x < THRESH else 1 for x in Z])
Z = Z.reshape(xx.shape)
ax.contourf(xx, yy, Z, alpha=.3, cmap=plt.cm.Paired)

T = intermediateModel.predict(layerVals)
T = np.array([LABEL if x < THRESH else 1 for x in T])
T = T.reshape(layerVals[:, 0].shape)
ax.scatter(layerVals[:, 0], layerVals[:, 1], color=colors[T].tolist(), s=100, alpha=.9)
ax.set_xlabel("h0")
ax.set_ylabel("h1")
plt.show()

# create a mesh to plot in
h = .02  # step size in the mesh
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
meshData = np.c_[xx.ravel(), yy.ravel()]

fig, ax = plt.subplots()
Z = model.predict(meshData)
Z = np.array([LABEL if x < THRESH else 1 for x in Z])
Z = Z.reshape(xx.shape)
ax.contourf(xx, yy, Z, alpha=.3, cmap=plt.cm.Paired)
ax.axis('off')

# Plot also the training points
T = model.predict(X)
T = np.array([LABEL if x < THRESH else 1 for x in T])
T = T.reshape(X[:,0].shape)
ax.scatter(X[:, 0], X[:, 1], color=colors[T].tolist(), s=100, alpha=.9)
plt.show()
