#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
from numpy.linalg import inv, norm
import matplotlib.pyplot as plt



def project_onto_feasible_set(X, c, d):
    """
    If X is not in the feasible set defined by c^T X <= d, project it onto the set.
    """
    if np.dot(c.T, X) > d:
        # Project onto the hyperplane defined by c^T X = d
        return X - ((np.dot(c.T, X) - d) / np.dot(c.T, c)) * c
    return X


# Constants
n = 50  # Dimension of X and a_i
m = 500  # Number of a_i and b_i
num_samples = 10000  # Number of X samples
variance = 4  # Variance of Gaussian distribution

# Generating the matrices and vectors
np.random.seed(0)  # For reproducibility
ai = np.random.normal(0, np.sqrt(variance), (m, n, 1))  # m ai matrices (nx1)
bi = np.random.normal(0, np.sqrt(variance), m)  # m bi scalars
c = np.random.normal(0, np.sqrt(variance), (n, 1))  # c matrix (nx1)
d = np.random.normal(0, np.sqrt(variance))  # d scalar

# Generate X samples
X_samples = np.linspace(-500, 500, num_samples).reshape(-1, 1)
X_samples = np.tile(X_samples, (1, n))

# Finding X_Feasible
X_feasible = []
for X in X_samples:
    X = X.reshape(-1, 1)
    if np.dot(c.T, X) <= d:
        X_feasible.append(X)

# Calculate f(X) for each X in X_Feasible and find the index i with the max |ai^TX + bi|
f_X = []
max_indices = []
for X in X_feasible:
    values = [abs(np.dot(ai[i].T, X) + bi[i]) for i in range(m)]
    max_value = max(values)
    max_index = values.index(max_value)
    f_X.append(max_value)
    max_indices.append(max_index)

# Show the results
len(X_feasible), f_X[:5], max_indices[:5]  # Displaying first 5 for brevity



# Recalculating f(X) for each X in X_Feasible and finding the index i with the max |ai^TX + bi|
f_X_values = []
max_indices_per_X = []

for X in X_feasible:
    values = [abs(np.dot(ai[i].T, X) + bi[i]) for i in range(m)]
    max_value = max(values)
    max_index = values.index(max_value)
    f_X_values.append(max_value[0][0])  # Extracting scalar value
    max_indices_per_X.append(max_index)

# Show the results for the first 5 X_feasible for brevity
f_X_values[:5], max_indices_per_X[:5]



# Checking the specific values for the 9th, 10th, 11th, and 12th X_feasible
indices_to_check = [8, 9, 10, 11]  # 9th, 10th, 11th, 12th indices in zero-based indexing

# Extracting the specific a_222 and b_222
a_222 = ai[222]
b_222 = bi[222]

# Calculating |a_222^TX + b_222| for the specified X matrices
values_at_222 = [abs(np.dot(a_222.T, X_feasible[i]) + b_222)[0][0] for i in indices_to_check]

# Calculating the maximum |ai^TX + bi| for the specified X matrices and their indices
max_values = []
max_indices = []
for i in indices_to_check:
    X = X_feasible[i]
    values = [abs(np.dot(ai[j].T, X) + bi[j]) for j in range(m)]
    max_value = max(values)
    max_index = values.index(max_value)
    max_values.append(max_value[0][0])  # Extracting scalar value
    max_indices.append(max_index)

#values_at_222, max_values, max_indices



def efficient_subgradient(ai, bi, X):
    """Calculate the subgradient of f(X) efficiently by finding the max contributor only."""
    contributions = [np.abs(np.dot(ai[i].T, X) + bi[i]) for i in range(m)]
    max_index = np.argmax(contributions)
    subgrad = np.sign(np.dot(ai[max_index].T, X) + bi[max_index]) * ai[max_index]
    return subgrad, max_index

# Reducing the number of iterations for quick testing
iterations = 50

# Initialize X within the feasible set
X_initial = np.random.normal(0, np.sqrt(variance), (n, 1))  # Random initial X
X_initial = project_onto_feasible_set(X_initial, c, d)  # Project onto feasible set

# Containers for the history of X and f(X) values
X_history_constant = [X_initial.copy()]
X_history_decreasing = [X_initial.copy()]
f_X_history_constant = []
f_X_history_decreasing = []

# Perform the projected sub-gradient method with both step size sequences
for k in range(1, iterations + 1):
    # Calculate step sizes for both sequences
    alpha_constant = 0.1 / np.sqrt(k)
    alpha_decreasing = 1 / k
    
    # Calculate subgradient efficiently
    gk_constant, _ = efficient_subgradient(ai, bi, X_history_constant[-1])
    gk_decreasing, _ = efficient_subgradient(ai, bi, X_history_decreasing[-1])
    
    # Update X for both step size sequences
    X_next_constant = X_history_constant[-1] - alpha_constant * gk_constant
    X_next_decreasing = X_history_decreasing[-1] - alpha_decreasing * gk_decreasing
    
    # Project back onto the feasible set
    X_next_constant = project_onto_feasible_set(X_next_constant, c, d)
    X_next_decreasing = project_onto_feasible_set(X_next_decreasing, c, d)
    
    # Store the updated X
    X_history_constant.append(X_next_constant)
    X_history_decreasing.append(X_next_decreasing)
    
    # Calculate and store current f(X)
    f_X_history_constant.append(np.max([np.abs(np.dot(ai[i].T, X_next_constant) + bi[i]) for i in range(m)]))
    f_X_history_decreasing.append(np.max([np.abs(np.dot(ai[i].T, X_next_decreasing) + bi[i]) for i in range(m)]))

# Final values after iterations
final_f_X_constant = f_X_history_constant[-1]
final_f_X_decreasing = f_X_history_decreasing[-1]
final_f_X_constant, final_f_X_decreasing



def subgradient(ai, bi, X):
    """
    Compute the subgradient of the function f(X) at X.
    f(X) = max_i |a_i^T X + b_i|
    The subgradient at X is a_i for the i that maximizes |a_i^T X + b_i|.
    """
    # Compute all values of |ai^T X + bi|
    values = [np.dot(ai[i].T, X) + bi[i] for i in range(len(ai))]
    # Get the index of the max value
    max_index = np.argmax(np.abs(values))
    # The subgradient is the a_i associated with the max value, considering the sign of the inner product
    subgrad = ai[max_index] if values[max_index] > 0 else -ai[max_index]
    return subgrad, max_index



def projected_subgradient_method(ai, bi, c, d, step_size_rule, num_iterations=1000):
    # Initialize X with a feasible value (we can start from the center of the feasible set for simplicity)
    X = np.zeros((n, 1))
    if np.dot(c.T, X) > d:
        X = project_onto_feasible_set(X, c, d)
    
    f_X_history = []  # Store the history of f(X) values
    X_history = []  # Store the history of X values
    index_history = []  # Store the history of indices which give max |ai^T X + bi|

    for k in range(1, num_iterations + 1):
        # Calculate the step size based on the given rule
        alpha_k = 0.1 / np.sqrt(k) if step_size_rule == "sqrt" else 1 / k

        # Compute the subgradient at the current X
        subgrad, max_index = subgradient(ai, bi, X)

        # Update X in the direction opposite to the subgradient
        X = X - alpha_k * subgrad

        # Project X back onto the feasible set if necessary
        X = project_onto_feasible_set(X, c, d)

        # Calculate the value of f(X)
        f_X = max([abs(np.dot(ai[i].T, X) + bi[i]) for i in range(m)])
        
        # Store the history
        f_X_history.append(f_X[0][0])
        X_history.append(X)
        index_history.append(max_index)

    return f_X_history, X_history, index_history

# Run the projected subgradient method with both step size rules
f_X_history_sqrt, X_history_sqrt, index_history_sqrt = projected_subgradient_method(
    ai, bi, c, d, step_size_rule="sqrt")
f_X_history_k, X_history_k, index_history_k = projected_subgradient_method(
    ai, bi, c, d, step_size_rule="k")

f_X_history_sqrt[-1], f_X_history_k[-1]  # Display the last values of f(X) for both step size rules


# In[46]:


import matplotlib.pyplot as plt

def norm_difference(X_history):
    """
    Calculate the Euclidean norm of the difference between consecutive X values.
    """
    return [np.linalg.norm(X_history[k] - X_history[k - 1]) for k in range(1, len(X_history))]

# Calculate the norm differences for both step size rules
norm_diff_sqrt = norm_difference(X_history_sqrt)
norm_diff_k = norm_difference(X_history_k)

# Plotting the value of the cost function over iterations
plt.figure(figsize=(14, 7))

# Cost function plot for step size rule sqrt(k)
plt.subplot(1, 2, 1)
plt.plot(f_X_history_sqrt, label='Cost Function Value ($f(X)$)')
plt.plot(norm_diff_sqrt, label='Norm Difference ($||X_{k+1} - X_k||_2$)')
plt.title('Step Size Rule $\\alpha_k = 0.1 / \\sqrt{k}$')
plt.xlabel('Iteration')
plt.ylabel('Value')
plt.legend()

# Cost function plot for step size rule 1/k
plt.subplot(1, 2, 2)
plt.plot(f_X_history_k, label='Cost Function Value ($f(X)$)')
plt.plot(norm_diff_k, label='Norm Difference ($||X_{k+1} - X_k||_2$)')
plt.title('Step Size Rule $\\alpha_k = 1 / k$')
plt.xlabel('Iteration')
plt.ylabel('Value')
plt.legend()

plt.tight_layout()
plt.show()



# In[43]:


# Redefine the simulation to find the minimum f(X) for all step size rules

def find_minimum_f_X(ai, bi, c, d, num_samples=10000, num_iterations=1000):
    # Generate more diverse X samples uniformly distributed in the range (-1000, 1000)
    X_samples = np.random.uniform(-1000, 1000, (num_samples, n, 1))

    # Finding X_Feasible
    X_feasible = [X for X in X_samples if np.dot(c.T, X) <= d]

    # Run the projected subgradient method for different step size rules and track the minimum f(X)
    min_f_X_results = {}
    for step_size_rule in ['sqrt', 'k', 'constant']:
        # Initialize X with a random feasible value
        X = X_feasible[np.random.randint(len(X_feasible))]
        
        min_f_X = float('inf')  # Initialize minimum f(X) as infinity

        for k in range(1, num_iterations + 1):
            # Calculate the step size based on the given rule
            if step_size_rule == 'sqrt':
                alpha_k = 0.1 / np.sqrt(k)
            elif step_size_rule == 'k':
                alpha_k = 1 / k
            else:  # constant
                alpha_k = 0.1

            # Compute the subgradient at the current X
            subgrad, _ = subgradient(ai, bi, X)

            # Update X in the direction opposite to the subgradient with the step size
            X = X - alpha_k * subgrad

            # Project X back onto the feasible set if necessary
            X = project_onto_feasible_set(X, c, d)

            # Calculate the value of f(X)
            f_X = max([abs(np.dot(ai[i].T, X) + bi[i]) for i in range(m)])
            
            # Update the minimum f(X) if the current f(X) is lower
            if f_X < min_f_X:
                min_f_X = f_X
        
        min_f_X_results[step_size_rule] = min_f_X[0][0]  # Extract scalar value
    
    return min_f_X_results, X_feasible

# Perform the simulation to find the minimum f(X) with diverse X samples
min_f_X_results, X_feasible = find_minimum_f_X(ai, bi, c, d)

min_f_X_results


# In[40]:


# Redefine the projected subgradient method to include a constant step size
def projected_subgradient_method_constant_step(ai, bi, c, d, alpha_k, num_iterations=1000):
    # Initialize X with a feasible value (we can start from the center of the feasible set for simplicity)
    X = np.zeros((n, 1))
    if np.dot(c.T, X) > d:
        X = project_onto_feasible_set(X, c, d)
    
    f_X_history = []  # Store the history of f(X) values
    X_history = []  # Store the history of X values
    index_history = []  # Store the history of indices which give max |ai^T X + bi|

    for k in range(1, num_iterations + 1):
        # Compute the subgradient at the current X
        subgrad, max_index = subgradient(ai, bi, X)

        # Update X in the direction opposite to the subgradient with a constant step size
        X = X - alpha_k * subgrad

        # Project X back onto the feasible set if necessary
        X = project_onto_feasible_set(X, c, d)

        # Calculate the value of f(X)
        f_X = max([abs(np.dot(ai[i].T, X) + bi[i]) for i in range(m)])
        
        # Store the history
        f_X_history.append(f_X[0][0])
        X_history.append(X)
        index_history.append(max_index)

    return f_X_history, X_history, index_history

# Constant step size
constant_alpha_k = 0.1  # Example constant step size

# Run the projected subgradient method with a constant step size
f_X_history_constant, X_history_constant, index_history_constant = projected_subgradient_method_constant_step(
    ai, bi, c, d, constant_alpha_k)

# Calculate the norm differences for the constant step size
norm_diff_constant = norm_difference(X_history_constant)

# Plotting the value of the cost function and norm difference over iterations for the constant step size
plt.figure(figsize=(7, 5))

plt.plot(f_X_history_constant, label='Cost Function Value ($f(X)$)')
plt.plot(norm_diff_constant, label='Norm Difference ($||X_{k+1} - X_k||_2$)')
plt.title('Constant Step Size $\\alpha_k = 0.1$')
plt.xlabel('Iteration')
plt.ylabel('Value')
plt.legend()

plt.tight_layout()
plt.show()


# In[34]:


# Redefine the projected subgradient method with the new X generation and constant step size
def projected_subgradient_method_full(ai, bi, c, d, step_size, num_samples=10000, num_iterations=1000):
    # Generate more diverse X samples uniformly distributed in the range (-1000, 1000)
    X_samples = np.random.uniform(-1000, 1000, (num_samples, n, 1))

    # Finding X_Feasible
    X_feasible = [X for X in X_samples if np.dot(c.T, X) <= d]

    # Initialize X with a random feasible value
    X = X_feasible[np.random.randint(len(X_feasible))]
    
    f_X_history = []  # Store the history of f(X) values
    X_history = []  # Store the history of X values
    norm_diff_history = []  # Store the history of norm differences

    for k in range(1, num_iterations + 1):
        # Compute the subgradient at the current X
        subgrad, _ = subgradient(ai, bi, X)

        # Update X in the direction opposite to the subgradient with the step size
        X_new = X - step_size * subgrad

        # Project X back onto the feasible set if necessary
        X_new = project_onto_feasible_set(X_new, c, d)

        # Calculate the value of f(X)
        f_X = max([abs(np.dot(ai[i].T, X_new) + bi[i]) for i in range(m)])
        
        # Calculate the norm difference
        norm_diff = np.linalg.norm(X_new - X)

        # Store the history
        f_X_history.append(f_X[0][0])
        norm_diff_history.append(norm_diff)
        X_history.append(X_new)
        
        # Update X for the next iteration
        X = X_new

    return f_X_history, X_history, norm_diff_history

# Example constant step size
constant_alpha_k = 0.1

# Run the projected subgradient method with the new X generation approach and constant step size
f_X_history_new, X_history_new, norm_diff_history_new = projected_subgradient_method_full(
    ai, bi, c, d, constant_alpha_k)

# Plotting the value of the cost function and norm difference over iterations for the new X generation
plt.figure(figsize=(7, 5))

plt.plot(f_X_history_new, label='Cost Function Value ($f(X)$)')
plt.plot(norm_diff_history_new, label='Norm Difference ($||X_{k+1} - X_k||_2$)')
plt.title('Diverse X Samples with Constant Step Size $\\alpha_k = 0.1$')
plt.xlabel('Iteration')
plt.ylabel('Value')
plt.legend()

plt.tight_layout()
plt.show()


# In[42]:


# Redefine the simulation to find the minimum f(X) for all step size rules

def find_minimum_f_X(ai, bi, c, d, num_samples=10000, num_iterations=1000):
    # Generate more diverse X samples uniformly distributed in the range (-1000, 1000)


    # Run the projected subgradient method for different step size rules and track the minimum f(X)
    min_f_X_results = {}
    for step_size_rule in ['sqrt', 'k', 'constant']:
        # Initialize X with a random feasible value
        X = X_feasible[np.random.randint(len(X_feasible))]
        
        min_f_X = float('inf')  # Initialize minimum f(X) as infinity

        for k in range(1, num_iterations + 1):
            # Calculate the step size based on the given rule
            if step_size_rule == 'sqrt':
                alpha_k = 0.1 / np.sqrt(k)
            elif step_size_rule == 'k':
                alpha_k = 1 / k
            else:  # constant
                alpha_k = 0.1

            # Compute the subgradient at the current X
            subgrad, _ = subgradient(ai, bi, X)

            # Update X in the direction opposite to the subgradient with the step size
            X = X - alpha_k * subgrad

            # Project X back onto the feasible set if necessary
            X = project_onto_feasible_set(X, c, d)

            # Calculate the value of f(X)
            f_X = max([abs(np.dot(ai[i].T, X) + bi[i]) for i in range(m)])
            
            # Update the minimum f(X) if the current f(X) is lower
            if f_X < min_f_X:
                min_f_X = f_X
        
        min_f_X_results[step_size_rule] = min_f_X[0][0]  # Extract scalar value
    
    return min_f_X_results, X_feasible

# Perform the simulation to find the minimum f(X) with diverse X samples
min_f_X_results, X_feasible = find_minimum_f_X(ai, bi, c, d)

min_f_X_results


# In[36]:


# Redefine the full simulation with more diverse X generation and apply it to all methods

def full_simulation_with_diverse_X(ai, bi, c, d, num_samples=10000, num_iterations=1000):
    # Generate more diverse X samples uniformly distributed in the range (-1000, 1000)
    X_samples = np.random.uniform(-1000, 1000, (num_samples, n, 1))

    # Finding X_Feasible
    X_feasible = [X for X in X_samples if np.dot(c.T, X) <= d]

    # Run the projected subgradient method for different step size rules
    results = {}
    for step_size_rule in ['sqrt', 'k', 'constant']:
        # Initialize X with a random feasible value
        X = X_feasible[np.random.randint(len(X_feasible))]
        
        f_X_history = []  # Store the history of f(X) values
        X_history = []  # Store the history of X values
        norm_diff_history = []  # Store the history of norm differences

        for k in range(1, num_iterations + 1):
            # Calculate the step size based on the given rule
            if step_size_rule == 'sqrt':
                alpha_k = 0.1 / np.sqrt(k)
            elif step_size_rule == 'k':
                alpha_k = 1 / k
            else:  # constant
                alpha_k = 0.1

            # Compute the subgradient at the current X
            subgrad, _ = subgradient(ai, bi, X)

            # Update X in the direction opposite to the subgradient with the step size
            X_new = X - alpha_k * subgrad

            # Project X back onto the feasible set if necessary
            X_new = project_onto_feasible_set(X_new, c, d)

            # Calculate the value of f(X)
            f_X = max([abs(np.dot(ai[i].T, X_new) + bi[i]) for i in range(m)])
            
            # Calculate the norm difference
            norm_diff = np.linalg.norm(X_new - X)

            # Store the history
            f_X_history.append(f_X[0][0])
            norm_diff_history.append(norm_diff)
            X_history.append(X_new)
            
            # Update X for the next iteration
            X = X_new
        
        results[step_size_rule] = (f_X_history, norm_diff_history)
    
    return results, X_feasible

# Perform the full simulation with diverse X samples
results, X_feasible = full_simulation_with_diverse_X(ai, bi, c, d)

# Plotting the results for all step size rules
plt.figure(figsize=(21, 5))

for idx, (step_size_rule, (f_X_history, norm_diff_history)) in enumerate(results.items()):
    plt.subplot(1, 3, idx + 1)
    plt.plot(f_X_history, label='Cost Function Value ($f(X)$)')
    plt.plot(norm_diff_history, label='Norm Difference ($||X_{k+1} - X_k||_2$)')
    plt.title(f'Step Size Rule: {step_size_rule}')
    plt.xlabel('Iteration')
    plt.ylabel('Value')
    plt.legend()

plt.tight_layout()
plt.show()

# For brevity, return the final f(X) values for each step size rule
{rule: history[0][-1] for rule, history in results.items()}


# In[38]:


# Redefine the simulation to find the minimum f(X) for all step size rules

def find_minimum_f_X(ai, bi, c, d, num_samples=10000, num_iterations=1000):
    # Generate more diverse X samples uniformly distributed in the range (-1000, 1000)
    X_samples = np.random.uniform(-1000, 1000, (num_samples, n, 1))

    # Finding X_Feasible
    X_feasible = [X for X in X_samples if np.dot(c.T, X) <= d]

    # Run the projected subgradient method for different step size rules and track the minimum f(X)
    min_f_X_results = {}
    for step_size_rule in ['sqrt', 'k', 'constant']:
        # Initialize X with a random feasible value
        X = X_feasible[np.random.randint(len(X_feasible))]
        
        min_f_X = float('inf')  # Initialize minimum f(X) as infinity

        for k in range(1, num_iterations + 1):
            # Calculate the step size based on the given rule
            if step_size_rule == 'sqrt':
                alpha_k = 0.1 / np.sqrt(k)
            elif step_size_rule == 'k':
                alpha_k = 1 / k
            else:  # constant
                alpha_k = 0.1

            # Compute the subgradient at the current X
            subgrad, _ = subgradient(ai, bi, X)

            # Update X in the direction opposite to the subgradient with the step size
            X = X - alpha_k * subgrad

            # Project X back onto the feasible set if necessary
            X = project_onto_feasible_set(X, c, d)

            # Calculate the value of f(X)
            f_X = max([abs(np.dot(ai[i].T, X) + bi[i]) for i in range(m)])
            
            # Update the minimum f(X) if the current f(X) is lower
            if f_X < min_f_X:
                min_f_X = f_X
        
        min_f_X_results[step_size_rule] = min_f_X[0][0]  # Extract scalar value
    
    return min_f_X_results, X_feasible

# Perform the simulation to find the minimum f(X) with diverse X samples
min_f_X_results, X_feasible = find_minimum_f_X(ai, bi, c, d)

min_f_X_results


# In[ ]:




