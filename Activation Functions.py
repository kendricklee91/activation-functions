import matplotlib.pyplot as plt
import numpy as np

x = np.arange(-9, 9, 0.01)

def identity_func(x):
    return x

def linear_func(x):
    return 1.5 * x + 1 # slope : 1.5, y_intercept : 1

def step_func(x):
    return (x >= 0) * 1

def sigmoid_func(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative_func(x):
    return sigmoid_func(x) * (1 - sigmoid_func(x))

def tanh_func(x):
    return np.tanh(x)

def relu_func(x):
    return x * (x > 0)

def relu_derivative_func(x):
    return 1 * (x > 0)

def leakyrelu_func(x):
    return (x >= 0) * x + (x < 0) * 0.01 * x # alpha value : 0.01

def gaussian_func(x):
    return np.exp(-x * x)

fig = plt.figure(figsize=  (10, 6))
ax1 = fig.add_subplot(111)

ax1.plot(x, identity_func(x), linestyle = '--', label = 'identity')
ax1.plot(x, linear_func(x), linestyle = '--', label = 'linear')
ax1.plot(x, step_func(x), linestyle = '--', label = 'step')
ax1.legend(loc = 2) # '2' mean is upper left of label location

fig.savefig('fig1.png', dpi = 300)
plt.show()

#######################################################################################

fig2 = plt.figure(figsize=  (10, 6))
ax2 = fig2.add_subplot(111)

ax2.plot(x, sigmoid_func(x), linestyle = '--', label = 'sigmoid')
ax2.plot(x, sigmoid_derivative_func(x), linestyle = '--', label = 'sigmoid_derivative')
ax2.plot(x, tanh_func(x), linestyle = '--', label = 'tanh')
ax2.legend(loc = 2) # '2' mean is upper left of label location

fig2.savefig('fig2.png', dpi = 300)
plt.show()

#######################################################################################

fig3 = plt.figure(figsize=  (10, 6))
ax3 = fig3.add_subplot(111)

ax3.plot(x, relu_func(x), linestyle = '--', label = 'relu')
ax3.plot(x, relu_derivative_func(x), linestyle = '--', label = 'relu_derivative')
ax3.plot(x, gaussian_func(x), linestyle = '--', label = 'gaussian')
ax3.legend(loc = 2) # '2' mean is upper left of label location

fig3.savefig('fig3.png', dpi = 300)
plt.show()

#######################################################################################

fig4 = plt.figure(figsize=  (10, 6))
ax4 = fig4.add_subplot(111)

ax4.plot(x, leakyrelu_func(x), linestyle = '--', label = 'leaky_relu')
ax4.legend(loc = 2)

fig4.savefig('fig4.png', dpi = 300)
plt.show()