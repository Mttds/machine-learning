import copy
import math
import numpy as np
import matplotlib.pyplot as plt
import os
import ctypes
import time

# Load the shared library
gradient_descent_lib = ctypes.CDLL('./bin/gradient_descent.so')

# Define the function signature
gradient_descent_lib.gradient_descent.argtypes = [
    ctypes.POINTER(ctypes.c_double),  # x array pointer
    ctypes.POINTER(ctypes.c_double),  # y array pointer
    ctypes.c_double,  # w_in
    ctypes.c_double,  # b_in
    ctypes.c_int,     # m
    ctypes.c_double,  # alpha
    ctypes.c_int,     # num_iters
    ctypes.POINTER(ctypes.c_double),  # w_final
    ctypes.POINTER(ctypes.c_double)   # b_final
]

def msg(s):
    print(f"[INFO] - {s}")

def load_data(filename):
    data = np.loadtxt(filename, delimiter=',')
    X = data[:,0]
    y = data[:,1]
    return X, y

def compute_cost(x, y, w, b): 
    """
    Computes the cost function for linear regression.
    
    Args:
        x (ndarray): Shape (m,) Input to the model (Population of cities) 
        y (ndarray): Shape (m,) Label (Actual profits for the cities)
        w, b (scalar): Parameters of the model
    
    Returns
        total_cost (float): The cost of using w,b as the parameters for linear regression
               to fit the data points in x and y
    """
    # number of training examples
    m = x.shape[0] 

    total_cost = 0

    for i in range(m):
        ith_f_wb_x = w * x[i] + b
        ith_cost = math.pow(ith_f_wb_x - y[i], 2)
        total_cost += ith_cost

    total_cost = total_cost * 1 / (2*m)
    return total_cost

def compute_gradient(x, y, w, b): 
    """
    Computes the gradient for linear regression 
    Args:
      x (ndarray): Shape (m,) Input to the model (Population of cities) 
      y (ndarray): Shape (m,) Label (Actual profits for the cities)
      w, b (scalar): Parameters of the model  
    Returns
      dj_dw (scalar): The gradient of the cost w.r.t. the parameters w
      dj_db (scalar): The gradient of the cost w.r.t. the parameter b     
     """

    # Number of training examples
    m = x.shape[0]

    dj_dw = 0
    dj_db = 0

    for i in range(m):
        ith_f_wb = w * x[i] + b
        ith_dj_db = ith_f_wb - y[i]
        ith_dj_dw = (ith_f_wb - y[i]) * x[i]
        dj_db += ith_dj_db
        dj_dw += ith_dj_dw
        
    dj_db = 1/m * dj_db
    dj_dw = 1/m * dj_dw

    return dj_dw, dj_db

def gradient_descent(x, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters): 
    """
    Performs batch gradient descent to learn theta. Updates theta by taking 
    num_iters gradient steps with learning rate alpha

    Args:
      x :    (ndarray): Shape (m,)
      y :    (ndarray): Shape (m,)
      w_in, b_in : (scalar) Initial values of parameters of the model
      cost_function: function to compute cost
      gradient_function: function to compute the gradient
      alpha : (float) Learning rate
      num_iters : (int) number of iterations to run gradient descent
    Returns
      w : (ndarray): Shape (1,) Updated values of parameters of the model after
          running gradient descent
      b : (scalar)                Updated value of parameter of the model after
          running gradient descent
    """
    # number of training examples
    m = len(x)

    # An array to store cost J and w's at each iteration — primarily for graphing later
    J_history = []
    w_history = []
    w = copy.deepcopy(w_in)  #avoid modifying global w within function
    b = b_in

    for i in range(num_iters):
        # Calculate the gradient and update the parameters
        dj_dw, dj_db = gradient_function(x, y, w, b)  

        # Update Parameters using w, b, alpha and gradient
        w = w - alpha * dj_dw               
        b = b - alpha * dj_db               

        # Save cost J at each iteration
        if i<100000: # prevent resource exhaustion 
            cost =  cost_function(x, y, w, b)
            J_history.append(cost)

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters/10) == 0:
            w_history.append(w)
            print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.2f}   ")
        
    return w, b, J_history, w_history # return w and J,w history for graphing

def restaurants_profit_model():
    # load the dataset
    # 1 feature x1 is the city population
    # the target y is the profit for the restaurant in the corresponding city
    x_train, y_train = load_data(os.path.join("data", "restaurants.csv"))
    m = len(x_train)

    # Convert data to ctypes pointers
    x_train_array = x_train.flatten('C').astype(ctypes.c_double)
    y_train_array = y_train.flatten('C').astype(ctypes.c_double)
    x_train_ptr = x_train_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    y_train_ptr = y_train_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    w_final = ctypes.c_double()
    b_final = ctypes.c_double()

    # print x_train
    msg(f"Type of x_train: {type(x_train)}")
    msg(f"First five elements of x_train are:\n{x_train[:5]}")

    # print y_train
    msg(f"Type of y_train: {type(y_train)}")
    msg(f"First five elements of y_train are:\n{y_train[:5]}")
    msg(f'The shape of x_train is: {x_train.shape}')
    msg(f'The shape of y_train is: {y_train.shape}')
    msg(f'Number of training examples (m): {len(x_train)}')

    # plot data
    plt.scatter(x_train, y_train, marker='x', c='r') 

    # Set the title
    plt.title("Profits vs. Population per city")
    # Set the y-axis label
    plt.ylabel('Profit in $10,000')
    # Set the x-axis label
    plt.xlabel('Population of City in 10,000s')
    plt.show()

    # initialize fitting parameters. run gradient descent
    initial_w = 0.
    initial_b = 0.

    # gradient descent settings
    iterations = 950000
    alpha = 0.01

    msg(f"Gradient descent start [Python] {time.time()}")
    start = time.time()
    w,b,_,_ = gradient_descent(
        x_train ,y_train, initial_w, initial_b, 
        compute_cost, compute_gradient, alpha, iterations
    )
    end = time.time()
    msg(f"Gradient descent end [Python] {time.time()}, Elapsed = {end - start}")
    msg(f"w,b found by gradient descent: {w}, {b}")

    msg(f"Gradient descent start [C OpenMP] {time.time()}")
    start = time.time()
    gradient_descent_lib.gradient_descent(
        x_train_ptr, y_train_ptr, initial_w, initial_b,
        m, alpha, iterations, ctypes.byref(w_final), ctypes.byref(b_final)
    )
    end = time.time()
    msg(f"Gradient descent end [C OpenMP] {time.time()}, Elapsed = {end - start}")

    w = w_final.value # c_double to python float
    b = b_final.value # c_double to python float
    msg(f"w,b found by gradient descent: {w}, {b}")

    m = x_train.shape[0]
    predicted = np.zeros(m)

    for i in range(m):
        predicted[i] = w * x_train[i] + b

    # Plot the linear fit
    plt.plot(x_train, predicted, c = "b")

    # Create a scatter plot of the data. 
    plt.scatter(x_train, y_train, marker='x', c='r') 

    # Set the title
    plt.title("Profits vs. Population per city")
    # Set the y-axis label
    plt.ylabel('Profit in $10,000')
    # Set the x-axis label
    plt.xlabel('Population of City in 10,000s')
    plt.show()

    p = 3.5 * w + b
    msg('For population = 35,000, we predict a profit of $%.2f' % (p*10000))

    p = 7.0 * w + b
    msg('For population = 70,000, we predict a profit of $%.2f' % (p*10000))

if __name__ == '__main__':
    restaurants_profit_model()
 
