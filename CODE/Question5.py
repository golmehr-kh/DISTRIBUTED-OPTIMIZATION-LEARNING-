#!/usr/bin/env python
# coding: utf-8

# # Question5: suggesting another method: Stochastic gradient descent

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt

def stochastic_gradient_descent(max_iterations, threshold, w0,
                                cost_func, gradient_func,
                                learning_rate, batch_size):
    
    w = w0
    w_history = w0 
    f_history = cost_func(w)
    i = 0
    diff = 0.1
    
    while i < max_iterations and diff > threshold:
        # Randomly select a subset for SGD
        random_indices = np.random.choice(len(w), size=batch_size, replace=False)
        w_batch = w[random_indices]
        
        # Compute the gradient using the selected subset
        delta_w = learning_rate * gradient_func(cost_func, w_batch)
        w = w - delta_w
        
        # Store the history of w and f
        w_history = np.vstack((w_history, w))
        f_history = np.vstack((f_history, cost_func(w)))
        
        # Update iteration number and diff between successive values
        # of the objective function
        i += 1
        diff = np.absolute(f_history[-1] - f_history[-2])
        print(f"Iteration {i}: Cost {cost_func(w)}, Weight {w}")
        
    # For 2-D Energy-Iteration Plot:
    plt.plot(f_history, marker='o', color='blue', markerfacecolor='red')
    plt.xlabel("Iteration", size=14)
    plt.ylabel("Energy (cost)", size=14)
    plt.show()
    
    # For 3-D trajectory plot:
    ax = plt.axes(projection='3d')
    zz = f_history
    z = np.array(zz.flatten())
    xx = w_history[:, 0]
    yy = w_history[:, 1]
    weightsplot = np.array([xx.flatten(), yy.flatten()])
    x = weightsplot[0]
    y = weightsplot[1]
    ax.plot3D(x, y, z, 'green', marker='o', markerfacecolor='yellow')
    ax.set_title('3D line plot')
    plt.xlabel(" W1", size=14)
    plt.ylabel(" W2", size=14)
    plt.show()

def gradd (f, x, h=1e-5):
    
    x = np.array(x)  
    grad = np.zeros_like(x)  # Initialize gradient array

    # Iterate over all dimensions
    for i in range(len(x)):
        x_step = np.copy(x)
        # Forward difference method
        x_step[i] += h
        f_x_plus_h = f(x_step)

        x_step[i] -= 2 * h
        f_x_minus_h = f(x_step)
        # Central difference for better accuracy
        grad[i] = (f_x_plus_h - f_x_minus_h) / (2 * h)

    return grad

# def gradd(cost_func, w_batch):
#     # Compute the gradient using the entire batch
#     return np.array([2 * (w_batch[0] - 2), 2 * (w_batch[1] - 3)])
# Your dataset (X, y) needs to be defined before calling the stochastic_gradient_descent function

# Initial parameters
max_iterations = 1000
threshold = 1e-5
x0 = [23, 37]
x_init = np.array(x0)
x_init = x_init.astype(np.float)

# Learning rate and batch size are important hyperparameters for SGD
learning_rate = 0.001
batch_size = 1

# Call the stochastic_gradient_descent function
stochastic_gradient_descent(max_iterations, threshold, x_init, f1, gradd, learning_rate, batch_size)

