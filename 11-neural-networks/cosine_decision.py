import math as m
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as datasets
from tensorflow import keras, math, random, stack
from tensorflow.keras import layers, initializers

# The model we want to learn is
#
#   if cos(4 * x[0]) + x[1] >= 0 then 1 else 0
# 
#       x[0] --- cos(4 * x[0]) 
#            \ /        \
#             X           output
#            / \        /
#       x[1] ---     x[1]
#

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
    x_0 = math.cos(x[...,0])
    x_1 = x[...,1]
    xnew = stack([x_0, x_1], axis = 1)
    return xnew

model = keras.models.Sequential()
model.add(layers.Dense(2, input_dim=2, use_bias=False, activation=custom_activation))
model.add(layers.Dense(1, activation=ACTIVATION))
#opt = keras.optimizers.Adam(learning_rate=.01)
model.compile(loss="binary_crossentropy") #, optimizer=opt)

centers = [[0, 0]]
t, _ = datasets.make_blobs(n_samples=7500, centers=centers, cluster_std=1,
                                random_state=0)

# CURVE
def generate_curve_data(t):
    # create some space between the classes
    X = np.array(list(filter(lambda x : m.cos(4*x[0]) - x[1] < -.5 or m.cos(4*x[0]) - x[1] > .5, t)))
    Y = np.array([1 if m.cos(4*x[0]) - x[1] >= 0 else LABEL for x in X])
    return X, Y

X, Y = generate_curve_data(t)

colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
colors = np.hstack([colors] * 20)

plt.scatter(X[:,0],X[:,1],color=colors[Y].tolist(), s=10, alpha=0.8)
plt.show()

history = model.fit(X, Y, batch_size=500, epochs=1000)

# Show the transformation of the input at the first hidden layer
layer = model.layers[0]
print(layer.get_config(), layer.get_weights())
keras_function = keras.backend.function([model.input], [layer.output])
layerVals = np.array(keras_function(X))[0]
plt.scatter(layerVals[:,0], layerVals[:, 1], color=colors[Y].tolist(), s=10, alpha=.8)
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
ax.axis('off')

T = intermediateModel.predict(layerVals)
T = np.array([LABEL if x < THRESH else 1 for x in T])
T = T.reshape(layerVals[:, 0].shape)
ax.scatter(layerVals[:, 0], layerVals[:, 1], color=colors[T].tolist(), s=10, alpha=0.9)
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
ax.scatter(X[:, 0], X[:, 1], color=colors[T].tolist(), s=10, alpha=0.9)
plt.show()
