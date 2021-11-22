import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

def custom_activation(x):
    return x**2

model = keras.models.Sequential()
model.add(layers.Dense(1, input_dim=1, use_bias=False, activation=custom_activation))
model.add(layers.Dense(1))
model.compile(loss="mean_squared_error")

log = np.array([1, 4])
X = -2.0 + 4.0 * np.random.random(100)
Y = log[0] + log[1] * X**2 + np.random.randn(100)

f, ax = plt.subplots()
ax.scatter(X, Y, color='tab:red', s=100, alpha=.9)
ax.set_xlabel("X")
ax.set_ylabel("Y")
plt.show()

f, ax = plt.subplots()
ax.scatter(X**2, Y, color='tab:red', s=100, alpha=.9)
ax.set_xlabel("X^2")
ax.set_ylabel("Y")
plt.show()

m = np.shape(X)[0]
X1 = np.array([np.ones(m),X]).T
beta = np.linalg.inv(X1.T @ X1) @ X1.T @ Y

xplot = np.linspace(-2,2,50)
yestplot = beta[0]+beta[1]*xplot
plt.plot(xplot,yestplot,lw=4)
plt.scatter(X,Y, color='tab:red', s=100)
plt.show()

X2 = np.array([np.ones(m),X**2]).T
beta = np.linalg.inv(X2.T @ X2) @ X2.T @ Y
yestplot = beta[0]+beta[1]*xplot**2

history = model.fit(X, Y, batch_size=50, epochs=2000)

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(xplot,yestplot,lw=4)
ax1.scatter(X,Y, color='tab:red', s=100)
ax1.set_title("Linear Regression")

meshData = np.linspace(-2,2,50)
Z = model.predict(meshData)

ax2.plot(meshData, Z, lw=4)
ax2.scatter(X, Y, color='tab:red', s=100, alpha=.9)
ax2.set_title("Neural Network")
plt.show()
