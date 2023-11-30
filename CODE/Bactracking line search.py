#!/usr/bin/env python
# coding: utf-8

# # Bactracking line search

# In[ ]:


def gradient_descent_BT(max_iterations, threshold, w0,
                     cost_func, gradient_func, plot_contours, 
                     alpha=0.5, beta=0.8):
    
    w = w0
    w_history = w0 
    f_history = cost_func(w)
    i = 0
    diff = 0.1
    
    while i < max_iterations and diff > threshold:
        # Compute the gradient
        gradient = gradient_func(cost_func, w)
        
        # Backtracking line search
        t = 1.0
        while cost_func(w - t * gradient) > cost_func(w) - alpha * t * np.dot(gradient, gradient):
            t *= beta
        
        # Update the weight
        w = w - t * gradient
        
        # Store the history of w and f
        w_history = np.vstack((w_history, w))
        f_history = np.vstack((f_history, cost_func(w)))
        
        # Update iteration number and diff between successive values
        # of the objective function
        i += 1
        diff = np.absolute(f_history[-1] - f_history[-2])
        print(f"Iteration {i}: Cost {cost_func(w)}, Weight {w}")
        
    if plot_contours:
        plot_quadratic_contours(cost_func, w_history)

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



#-------------- Plot Contours ---------------
    
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
    
  #-----------  Using Gradient Descent with Backtracking ----------
# -------------------- Uncomment each section to run the code ---------------
    
# x0 = [23, 37]
# x_init = np.array(x0)
# x_init = x_init.astype(np.float)
# gradient_descent_BT(1000, 0.0001 ,x_init, f1, gradd, True)


##################
# x0 = [1,2]
# x0 = np.array(x0)
# x0 = x0.astype(np.float)
# gradient_descent_BT(1000, 0.0001 ,x0, f2, gradd, False)
##################

# x0 = [1,2,2,2]
# x_init = np.array(x0)
# x_init = x_init.astype(np.float)
# gradient_descent_BT(500, 0.0001 ,x_init, f3, gradd , False)

