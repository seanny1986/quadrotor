import numpy as np
import matplotlib.pyplot as plt

def function(x):
    return x**3

def first_derivative(x, dx):
    return (function(x+dx)-function(x-dx))/(2*dx)

def second_derivative(x, dx):    
    return (first_derivative(x+dx, dx)-first_derivative(x-dx, dx))/(2*dx)

xs = np.arange(-2., 2., 0.01)
ys = function(xs)
dx = 1e-5
dydx = first_derivative(xs, dx)
ddydxx = second_derivative(xs, dx)

plt.plot(xs, ys)
plt.plot(xs, dydx)
plt.plot(xs, ddydxx)
plt.show()