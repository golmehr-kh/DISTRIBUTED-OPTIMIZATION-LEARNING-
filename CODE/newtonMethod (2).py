#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


def newton_hessian(max_iterations,threshold,w0h,
                     cost_func,gradient_func, hessian_func,
                     learning_rate, plot_contours):
    
    w = w0h
    w_history1 = w0h
    f_history1 = cost_func(w)
    delta_w = np.zeros(w.shape)
    i = 0
    diff = 0.1
    
    while  i < max_iterations and diff > threshold:
        delta_w = (learning_rate)*(np.matmul(hessian_func(cost_func,w), gradient_func(cost_func,w)))
        w = w - delta_w
        
        # store the history of w and f
        w_history1 = np.vstack((w_history1,w))
        f_history1 = np.vstack((f_history1,cost_func(w)))
        
        # update iteration number and diff between successive values
        # of objective function
        i+=1
        diff = np.absolute(f_history1[-1]-f_history1[-2])
        print(f"Iteration {i}: Cost {cost_func(w)}, Weight \ {w}")
        
    # Plot contours for the quadratic function
    if plot_contours:
        plot_quadratic_contours(cost_func, w_history1)
      
    #For 2-D Energy-Iteration Plot:
    plt.plot(f_history1, marker='o', color='blue',markerfacecolor='red')
    plt.xlabel("Iteration", size = 14)
    plt.ylabel("Energy (cost)", size = 14)
    plt.show()
    #For 3-D trajectory plot:
    ax = plt.axes(projection ='3d')
    zz = f_history1
    z = np.array(zz.flatten())
    xx = w_history1[ :,0]
    yy = w_history1[ :,1]
    weightsplot = np.array([xx.flatten(), yy.flatten()])
    x = weightsplot[0]
    y = weightsplot[1]
    ax.plot3D(x, y, z, 'green', marker='o',markerfacecolor='yellow')
    ax.set_title('3D line plot')
    plt.xlabel(" W1", size = 14)
    plt.ylabel(" W2", size = 14)
    plt.show()
    #return w_history1,f_history1

#------------------------------------------------


def plot_quadratic_contours(cost_func, trajectory):
    Q = np.array([[48, 12], [8, 8]])  # Adjust Q for your quadratic function
    b = np.array([[13], [23]])  # Adjust b for your quadratic function
    c = 4  # Adjust c for your quadratic function

    x = np.linspace(-60, 60, 100)
    y = np.linspace(-60, 60, 100)
    X, Y = np.meshgrid(x, y)
    XY = np.column_stack((X.ravel(), Y.ravel()))

    Z = cost_func(np.vstack(XY.transpose())).reshape(len(x)*len(x), len(y)*len(y))

    plt.contour (X, Y, Z[:len(Z):100,:len(Z):100], levels=30, cmap='viridis')
    plt.plot(trajectory[:, 0], trajectory[:, 1], marker='o', color='red', label='Gradient Descent Trajectory')
    plt.xlabel('W1')
    plt.ylabel('W2')
    plt.title('Quadratic Function Contour Plot')
    plt.legend()
    plt.show()
      
#--------------------------- Gradient (g) Function -----------------------------

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


# ------------------------ Define Hessian Function ------------------------------
def hessian(f, x, h=1e-5):
     
#     Parameters:
#     f (function): The function for which the Hessian is to be calculated.
#     x (array-like): The point at which the Hessian is to be calculated.
#     h (float): A small step size for calculating the finite difference.

    x = np.asarray(x, dtype=float)
    n = x.shape[0]
    hessian = np.zeros((n, n), dtype=float)
    fx = f(x)

    # Iterate over all entries in the Hessian matrix
    for i in range(n):
        for j in range(n):
            x[i] += h
            x[j] += h
            f_plus_plus = f(x)
            
            x[j] -= 2 * h
            f_plus_minus = f(x)
            
            x[i] -= 2 * h
            f_minus_minus = f(x)
            
            x[j] += 2 * h
            f_minus_plus = f(x)
            
            # Reset x to the original values
            x[i] += h
            x[j] -= h
            
            # Calculate second-order differences
            hessian[i, j] = (f_plus_plus - f_plus_minus - f_minus_plus + f_minus_minus) / (4 * h**2)

    return hessian

#-------------------------- Define the Cost functions ---------------------------------

# This code is adaptable for higher orders. Just you should apply proper changes in each of the following functions.
# for example, you have to change all the paramaters such as Q and q in the first functions.

        #---------first function----------
def f1(x):
    Q= [[48,12],[8,8]]
    Q = np.array(Q) 
    q=[[13],[23]]
    q= np.array(q)
    p=4
    return (0.5*np.dot(np.dot(x.transpose(),Q),x)+np.dot(q.transpose(),x)+p)


        #-------second function ---------
def f2(x):
    n=2 
    a = -2
    b =150
    
    i=1
    summ = 0
    for i in range(n-1):
        temp = b*np.power(((x[i]*x[i])-x[i-1]),2) + np.power((x[i-1]-a),2)
        summ += temp
    return(summ)

        #---------third funtion--------------
def f3(x):
    return( np.power((x[0]-10*x[1]),2) + 5*np.power((x[2]-x[3]),2) + np.power((x[1]-2*x[2]),4) + 10*np.power((x[0]-x[3]),4)) 



# ------------------------------- Using Newton's method -------------------------

# ---- Uncomment each section to run the code 

##### first function

# x0 = [23, 37]
# x_init = np.array(x0)
# x_init = w_init.astype(np.float)
# learning_rate = 0.0003
# newton_hessian(1000, 0.0001 ,x_init, f1, gradd, hessian, learning_rate, True)


##### second function

# x0 = [1,2]
# x_init = np.array(x0)
# x_init = x_init.astype(np.float)
# learning_rate = 0.001
# newton_hessian(1000, 0.0001 ,x_init, f2, gradd, hessian, learning_rate, False)

##### Third function

# x0 = [1,2,2,2]
# x_init = np.array(x0)
# x_init = x_init.astype(np.float)
# learning_rate = 1e-7
# newton_hessian(1500, 0.000000001 ,x_init, f2, gradd, hessian, learning_rate, False)


# In[19]:


# x0 = [3,-3]
# x_init = np.array(x0)
# x_init = x_init.astype(np.float)
# learning_rate = 0.0003
# newton_hessian(1000, 0.001 ,x_init, f1, gradd, hessian, learning_rate, True)


# In[22]:



# x0 = [21,5]
# x_init = np.array(x0)
# x_init = x_init.astype(np.float)
# learning_rate = 1e-8
# newton_hessian(1500, 0.00001 ,x_init, f2, gradd, hessian, learning_rate, False)


# In[47]:


# x0 = [1,0.8,0,1]
# x_init = np.array(x0)
# x_init = x_init.astype(np.float)
# learning_rate = 4e-5
# newton_hessian(6000, 0.00001 ,x_init, f3, gradd, hessian, learning_rate, False)

