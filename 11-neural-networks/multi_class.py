import math as m
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as datasets
from tensorflow import keras, math, random, stack
from tensorflow.keras import layers, initializers, utils

# 
#       x[0] --- h1 ---- out[0]
#            \ /    
#             X      
#            / \    
#       x[1] --- h2 ---- out[1]
#
# This is the base multi-class model

# modify this line
ACTIVATION = "softmax"
CLASSES = 3

# DONT MODIFY
# this is just for declaring
LABEL = None
THRESH = None
# Set random seed for reproducibility
np.random.seed(1)
random.set_seed(1)

def custom_activation(x):
    return x**2

model = keras.models.Sequential()
model.add(layers.Dense(3, input_dim=2, activation=custom_activation))
model.add(layers.Dense(CLASSES, activation=ACTIVATION))
model.compile(loss="categorical_crossentropy")

centers = [[0, 0]]
t, _ = datasets.make_blobs(n_samples=1500, centers=centers, cluster_std=2,
                                random_state=0)

# CIRCLES
def generate_circles_data(t):
    def label(x):
        if x[0]**2 + x[1]**2 >= 2 and x[0]**2 + x[1]**2 < 8:
            return 1
        if x[0]**2 + x[1]**2 >= 8:
            return 2
        return 0
    # create some space between the classes
    X = np.array(list(filter(lambda x : (x[0]**2 + x[1]**2 < 1.8 or x[0]**2 + x[1]**2 > 2.2) and (x[0]**2 + x[1]**2 < 7.8 or x[0]**2 + x[1]**2 > 8.2), t)))
    Y = np.array([label(x) for x in X])
    return X, Y

# LINES
def generate_lines_data(t):
    def label(x):
        if x[0] - x[1] >= 0 and x[0] - x[1] < 2:
            return 1
        if x[0] - x[1] >= 2:
            return 2
        return 0
    # create some space between the classes
    X = np.array(list(filter(lambda x : (x[0] - x[1] < -.2 or x[0] - x[1] > .2) and (x[0] - x[1] < 1.8 or x[0] - x[1] > 2.2), t)))
    Y = np.array([label(x) for x in X])
    return X, Y

X, Y = generate_circles_data(t)
# X, Y = generate_lines_data(t)

colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
colors = np.hstack([colors] * 20)

plt.scatter(X[:,0],X[:,1],color=colors[Y].tolist(), s=10, alpha=0.8)
plt.show()

history = model.fit(X, y=utils.to_categorical(Y), batch_size=500, epochs=1000)

# Print all the layer weights + biases
i = 0
for layer in model.layers:
    i += 1
    print("Layer ", i)
    print(layer.get_config(), layer.get_weights())

# Show the transformation of the input at the first hidden layer
layer = model.layers[0]
keras_function = keras.backend.function([model.input], [layer.output])
layerVals = np.array(keras_function(X))[0]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(layerVals[:,0], layerVals[:, 1], layerVals[:, 2], color=colors[Y].tolist(), s=10, alpha=.8)
plt.show()

# create a mesh to plot in
h = .1  # step size in the mesh
x_min, x_max = layerVals[:, 0].min() - .5, layerVals[:, 0].max() + 1
y_min, y_max = layerVals[:, 1].min() - .5, layerVals[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
meshData = np.c_[xx.ravel(), yy.ravel(), np.zeros(len(xx.ravel()))]

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh
fig, ax = plt.subplots()
layer = model.layers[-1]

intermediateModel = keras.models.Sequential()
intermediateModel.add(layers.Dense(CLASSES, input_dim=3, activation=ACTIVATION))
intermediateModel.compile(loss="categorical_crossentropy")
intermediateModel.layers[0].set_weights(layer.get_weights())

Zint = intermediateModel.predict(meshData)
Z = np.array([np.argmax(x) for x in Zint])
Z = Z.reshape(xx.shape)
ax.contourf(xx, yy, Z, alpha=.3) # plot in 2D
ax.axis('off')

Tint = intermediateModel.predict(layerVals)
T = np.array([np.argmax(x) for x in Tint])
T = T.reshape(layerVals[:, 0].shape)
ax.scatter(layerVals[:, 0], layerVals[:, 1], color=colors[T].tolist(), s=10, alpha=0.9) # plot in 2D
plt.show()

# create a mesh to plot in
h = .02  # step size in the mesh
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
meshData = np.c_[xx.ravel(), yy.ravel()]

fig, ax = plt.subplots()
Zint = model.predict(meshData)
Z = np.array([np.argmax(x) for x in Zint])
Z = Z.reshape(xx.shape)
ax.contourf(xx, yy, Z, alpha=.3)
ax.axis('off')

# Plot also the training points
Tint = model.predict(X)
T = np.array([np.argmax(x) for x in Tint])
T = T.reshape(X[:,0].shape)
ax.scatter(X[:, 0], X[:, 1], color=colors[T].tolist(), s=10, alpha=0.9)
plt.show()
