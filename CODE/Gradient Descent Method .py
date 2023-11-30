#!/usr/bin/env python
# coding: utf-8

# # Find the best learning rate for the quadratic function

# In[44]:


import numpy as np

def optimal_step_size(Q, gradient):
    numerator = np.dot(gradient.transpose(), gradient)
    denominator = np.dot(gradient.transpose(), np.dot(Q,gradient))
    alpha = numerator / denominator
    return alpha

# Example usage:
# Q is a positive definite matrix
Q = np.array([[48, 12], [8, 8]])
# Gradient vector
x = np.array([[23],[37]])
q =np.array([[13],[23]])
gradient = np.array(np.dot(Q,x)-q)

# Compute the optimal step size
alpha = optimal_step_size(Q, gradient)

print("Optimal Step Size:", alpha)


# In[2]:


import numpy as np
import matplotlib.pyplot as plt 
from mpl_toolkits import mplot3d

#---------------- Define Gradient Descent Function ----------------------------------

def gradient_descent(max_iterations,threshold,w0,
                     cost_func,gradient_func,
                     learning_rate,plot_contours):
    
    w = w0
    w_history = w0 
    f_history = cost_func(w)
    delta_w = np.zeros(w.shape)
    i = 0
    diff = 0.1
    
    while  i < max_iterations and diff > threshold:
        delta_w = (learning_rate)*(gradient_func(cost_func,w))
        w = w - delta_w
        
        # store the history of w and f
        w_history = np.vstack((w_history,w))
        f_history = np.vstack((f_history,cost_func(w)))
        
        # update iteration number and diff between successive values
        # of objective function
        i+=1
        diff = np.absolute(f_history[-1]-f_history[-2])
        print(f"Iteration {i}: Cost {cost_func(w)}, Weight \ {w}")
        
    # Plot contours for the quadratic function
    if plot_contours:
        plot_quadratic_contours(cost_func, w_history)
        
    #For 2-D Energy-Iteration Plot:
    plt.plot(f_history, marker='o', color='blue',markerfacecolor='red')
    plt.xlabel("Iteration", size = 14)
    plt.ylabel("Energy (cost)", size = 14)
    plt.show()
    
    #For 3-D trajectory plot:
    ax = plt.axes(projection ='3d')
    zz = f_history
    z = np.array(zz.flatten())
    xx = w_history[ :,0]
    yy = w_history[ :,1]
    weightsplot = np.array([xx.flatten(), yy.flatten()])
    x = weightsplot[0]
    y = weightsplot[1]
    ax.plot3D(x, y, z, 'green', marker='o' ,markerfacecolor='yellow')
    ax.set_title('3D line plot')
    plt.xlabel(" W1", size = 14)
    plt.ylabel(" W2", size = 14)
    plt.show()
    #return w_history,f_history

#------------ Energy (Cost) Function --------------------
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



#-------------- Gradient (g) Function ---------------

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
#------------------------------------------------


def plot_quadratic_contours(cost_func, trajectory):
    Q = np.array([[48, 12], [8, 8]])  # Adjust Q for your quadratic function
    b = np.array([[13], [23]])  # Adjust b for your quadratic function
    c = 4  # Adjust c for your quadratic function

    x = np.linspace(-40, 40, 100)
    y = np.linspace(-40, 40, 100)
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
    
    
#-----------  Using Gradient Descent ----------
# ---- Uncomment each section to run the code 

##### first function

# x0 = [23, 37]
# x_init = np.array(x0)
# x_init = x_init.astype(np.float)
# learning_rate = 0.019
# gradient_descent(1000, 0.0001 ,x_init, f1, gradd, learning_rate,True)


##### second function

# x0 = [1,2]
# x_init = np.array(x0)
# x_init = x_init.astype(np.float)
# learning_rate = 0.001
# gradient_descent(1000, 0.0001 ,x_init, f2, gradd,learning_rate,False)

##### Third function

# x0 = [1,2,2,2]
# x_init = np.array(x0)
# x_init = x_init.astype(np.float)
# learning_rate = 1e-4
# gradient_descent(1500, 0.000000001 ,x_init, f3, gradd, learning_rate,False)


# In[3]:


# x0 = [23, 37]
# x_init = np.array(x0)
# x_init = x_init.astype(np.float)
# learning_rate = 0.005
# gradient_descent(1000, 0.0001 ,x_init, f1, gradd, learning_rate,True)


# In[8]:


# x0 = [23, 37]
# x_init = np.array(x0)
# x_init = x_init.astype(np.float)
# learning_rate = 0.05
# gradient_descent(1000, 0.0001 ,x_init, f1, gradd, learning_rate,True)


# In[20]:


# x0 = [1,2]
# x_init = np.array(x0)
# x_init = x_init.astype(np.float)
# learning_rate = 0.005
# gradient_descent(1500, 0.0001 ,x_init, f2, gradd,learning_rate,False)


# In[21]:



# x0 = [1,2,2,2]
# x_init = np.array(x0)
# x_init = x_init.astype(np.float)
# learning_rate = 0.005
# gradient_descent(5000, 0.0001 ,x_init, f3, gradd, learning_rate, False)


# In[22]:


x0 = [1,2]
x0 = np.array(x0)
x0 = x0.astype(np.float)
learning_rate = 0.001

gradient_descent(4000, 0.0001 ,x0, f2, gradd,learning_rate, False)

