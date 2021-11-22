import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as datasets
from tensorflow import keras
from tensorflow.keras import layers

# The model we want to learn is
#
#   x[0] XOR x[1]
# 
#       x[0] --- h1 
#            \ /    \
#             X       output
#            / \    /
#       x[1] --- h2
#
# Suggestion is to use a the above architecture with the sigmoid activation function
#

# modify this line
ACTIVATION = "sigmoid"

# DONT MODIFY
# this is just for declaring
LABEL = None
THRESH = None

if ACTIVATION=="tanh":
    LABEL = -1
    THRESH = 0
if ACTIVATION=="sigmoid":
    LABEL = 0
    THRESH = 0.5

def custom_activation(x):
    return x**2

model = keras.models.Sequential()
model.add(layers.Dense(3, input_dim=2, activation=ACTIVATION))
model.add(layers.Dense(1, activation=ACTIVATION))
model.compile(loss="binary_crossentropy")

X = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]])
Y = np.array([x[0]^x[1] for x in X])

print(X)
print(Y)

colors = np.array([x for x in 'bg'])

plt.scatter(X[:,0],X[:,1],color=colors[Y].tolist(), s=50, alpha=0.8)
plt.show()

history = model.fit(X, Y, batch_size=4, epochs=10000) # need more epochs since less data

# Show the transformation of the input at the first hidden layer
layer = model.layers[0]
print(layer.get_config(), layer.get_weights())
keras_function = keras.backend.function([model.input], [layer.output])
layerVals = np.array(keras_function(X))[0]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(layerVals[:,0], layerVals[:, 1], layerVals[:, 2], color=colors[Y].tolist(), s=50, alpha=.8)
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
intermediateModel.add(layers.Dense(1, input_dim=3, activation=ACTIVATION))
intermediateModel.compile(loss="binary_crossentropy")
intermediateModel.layers[0].set_weights(layer.get_weights())

Z = intermediateModel.predict(meshData)
Z = np.array([LABEL if x < THRESH else 1 for x in Z])
Z = Z.reshape(xx.shape)
ax.contourf(xx, yy, Z, alpha=.3, cmap=plt.cm.Paired) # plot in 2D
ax.axis('off')

T = intermediateModel.predict(layerVals)
T = np.array([LABEL if x < THRESH else 1 for x in T])
T = T.reshape(layerVals[:, 0].shape)
ax.scatter(layerVals[:, 0], layerVals[:, 1], color=colors[T].tolist(), s=50, alpha=0.9) # plot in 2D
plt.show()


# create a mesh to plot in
h = .1  # step size in the mesh
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
meshData = np.c_[xx.ravel(), yy.ravel()]

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh
fig, ax = plt.subplots()
Zp = model.predict(meshData)
Z = np.array([np.array([LABEL]) if x < THRESH else np.array([1]) for x in Zp])
Z = Z.reshape(xx.shape)
ax.contourf(xx, yy, Z, alpha=.3, cmap=plt.cm.Paired)
ax.axis('off')

# Plot also the training points
T = model.predict(X)
T = np.array([LABEL if x < THRESH else 1 for x in T])
T = T.reshape(X[:,0].shape)
ax.scatter(X[:, 0], X[:, 1], color=colors[T].tolist(), s=50, alpha=0.9)
plt.show()