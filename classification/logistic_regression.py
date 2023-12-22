import copy
import math
import os
import numpy as np
import matplotlib.pyplot as plt

def msg(s):
    print(f"[INFO] - {s}")

def load_data(filename):
    data = np.loadtxt(filename, delimiter=',')
    X = data[:,:2]
    y = data[:,2]
    return X, y

def sigmoid(z):
    """
    sigmoid function
    """
    return 1/(1+np.exp(-z))

def compute_cost(X, y, w, b, lambda_, regularized):
    """
    Computes the cost over all examples
    Args:
      X : (ndarray Shape (m,n)) data, m examples by n features
      y : (ndarray Shape (m,))  target value 
      w : (ndarray Shape (n,))  values of parameters of the model      
      b : (scalar)              value of bias parameter of the model
      *argv : unused, for compatibility with regularized version below
    Returns:
      total_cost : (scalar) cost 
    """

    m, n = X.shape
    total_cost = 0.0

    for i in range(m):
        z_i = np.dot(X[i],w) + b
        f_wb = sigmoid(z_i)
        total_cost += (-y[i] * math.log(f_wb)) - (1 - y[i]) * math.log(1 - f_wb)
    
    reg_cost = 0.0
    if regularized:
        for j in range(n):
            reg_cost += w[j]**2
        reg_cost = reg_cost * lambda_/(2*m)
    
    return total_cost * 1/m + reg_cost

def compute_gradient(X, y, w, b, lambda_, regularized): 
    """
    Computes the gradient for logistic regression 
 
    Args:
      X : (ndarray Shape (m,n)) data, m examples by n features
      y : (ndarray Shape (m,))  target value 
      w : (ndarray Shape (n,))  values of parameters of the model      
      b : (scalar)              value of bias parameter of the model
      *argv : unused, for compatibility with regularized version below
    Returns
      dj_dw : (ndarray Shape (n,)) The gradient of the cost w.r.t. the parameters w. 
      dj_db : (scalar)             The gradient of the cost w.r.t. the parameter b. 
    """
    m, n = X.shape
    dj_dw = np.zeros(w.shape)
    dj_db = 0.

    for i in range(m):
        z_wb = np.dot(X[i], w) + b
        f_wb = sigmoid(z_wb)

        dj_db_i = f_wb - y[i]
        dj_db += dj_db_i
        
        for j in range(n):
            dj_dw[j] += (f_wb - y[i]) * X[i][j]
            if regularized:
                dj_dw[j] += lambda_/m * w[j]
            
    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_db, dj_dw

def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters, lambda_, regularized): 
    """
    Performs batch gradient descent to learn theta. Updates theta by taking 
    num_iters gradient steps with learning rate alpha
    
    Args:
      X :    (ndarray Shape (m, n) data, m examples by n features
      y :    (ndarray Shape (m,))  target value 
      w_in : (ndarray Shape (n,))  Initial values of parameters of the model
      b_in : (scalar)              Initial value of parameter of the model
      cost_function :              function to compute cost
      gradient_function :          function to compute gradient
      alpha : (float)              Learning rate
      num_iters : (int)            number of iterations to run gradient descent
      lambda_ : (scalar, float)    regularization constant
      
    Returns:
      w : (ndarray Shape (n,)) Updated values of parameters of the model after
          running gradient descent
      b : (scalar)                Updated value of parameter of the model after
          running gradient descent
    """
    
    # number of training examples
    m = len(X)
    
    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    w_history = []
    
    for i in range(num_iters):
        # Calculate the gradient and update the parameters
        dj_db, dj_dw = gradient_function(X, y, w_in, b_in, lambda_, regularized)   

        # Update Parameters using w, b, alpha and gradient
        w_in = w_in - alpha * dj_dw               
        b_in = b_in - alpha * dj_db              
       
        # Save cost J at each iteration
        if i < 100000: # prevent resource exhaustion 
            cost =  cost_function(X, y, w_in, b_in, lambda_, False)
            J_history.append(cost)

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters/10) == 0 or i == (num_iters-1):
            w_history.append(w_in)
            msg(f"Iteration {i:4}: Cost {float(J_history[-1]):8.2f}   ")
        
    return w_in, b_in, J_history, w_history

def predict(X, w, b): 
    """
    Predict whether the label is 0 or 1 using learned logistic
    regression parameters w
    
    Args:
      X : (ndarray Shape (m,n)) data, m examples by n features
      w : (ndarray Shape (n,))  values of parameters of the model      
      b : (scalar)              value of bias parameter of the model

    Returns:
      p : (ndarray (m,)) The predictions for X using a threshold at 0.5
    """
    # number of training examples
    m, n = X.shape   
    p = np.zeros(m)

    # Loop over each example
    for i in range(m):   
        z_wb = 0.0
        # Loop over each feature
        for j in range(n): 
            # Add the corresponding term to z_wb
            z_wb += X[i][j] * w[j]

        # Add bias term 
        z_wb += b

        # Calculate the prediction for this example
        f_wb = sigmoid(z_wb)

        # Apply the threshold (above for positive predition)
        p[i] = 1.0 if f_wb >= 0.5 else 0.0

    return p

def map_feature(X1, X2):
    """
    Feature mapping function to polynomial features    
    """
    X1 = np.atleast_1d(X1)
    X2 = np.atleast_1d(X2)
    degree = 6
    out = []
    for i in range(1, degree+1):
        for j in range(i + 1):
            out.append((X1**(i-j) * (X2**j)))
    return np.stack(out, axis=1)


def plot_data(X, y, pos_label="y=1", neg_label="y=0"):
    positive = y == 1
    negative = y == 0
    
    # Plot examples
    plt.plot(X[positive, 0], X[positive, 1], 'k+', label=pos_label)
    plt.plot(X[negative, 0], X[negative, 1], 'yo', label=neg_label)
    
    
def plot_decision_boundary(w, b, X, y):
    plot_data(X[:, 0:2], y)
    
    if X.shape[1] <= 2:
        plot_x = np.array([min(X[:, 0]), max(X[:, 0])])
        plot_y = (-1. / w[1]) * (w[0] * plot_x + b)
        
        plt.plot(plot_x, plot_y, c="b")
        
    else:
        u = np.linspace(-1, 1.5, 50)
        v = np.linspace(-1, 1.5, 50)
        
        z = np.zeros((len(u), len(v)))

        # Evaluate z = theta*x over the grid
        for i in range(len(u)):
            for j in range(len(v)):
                feature_vector = map_feature(u[i], v[j])
                if isinstance(feature_vector, np.ndarray):
                    feature_vector = feature_vector[0] # Extract a single element from the array

                z[i,j] = sigmoid(np.dot(feature_vector, w) + b)
        
        # important to transpose z before calling contour       
        z = z.T
        
        # Plot z = 0
        plt.contour(u,v,z, levels = [0.5], colors="g")


def university_admission_model():
    print("*********** University admission model ***********")
    # load dataset
    # first two columns are the feature (exam 1 and exam 2 scores)
    # last column is 1 for admitted and 0 for rejected
    X_train, y_train = load_data(os.path.join("data", "university_addmission.csv"))

    msg(f"First five elements in X_train are:\n{X_train[:5]}")
    msg(f"Type of X_train: {type(X_train)}")
    msg('The shape of X_train is: ' + str(X_train.shape))
    msg('The shape of y_train is: ' + str(y_train.shape))
    msg('We have m = %d training examples' % (len(y_train)))

    # Plot examples
    plot_data(X_train, y_train[:], pos_label="Admitted", neg_label="Not admitted")

    # Set the y-axis label
    plt.ylabel('Exam 2 score') 
    # Set the x-axis label
    plt.xlabel('Exam 1 score') 
    plt.legend(loc="upper right")
    plt.show()

    m, n = X_train.shape

    # Compute and display cost with w and b initialized to zeros
    initial_w = np.zeros(n)
    initial_b = 0.
    cost = compute_cost(X_train, y_train, initial_w, initial_b, 0, False)
    msg('Cost at initial w and b (zeros): {:.3f}'.format(cost))

    # Compute and display cost with non-zero w and b (random values)
    np.random.seed(1)
    initial_w = 0.01 * (np.random.rand(2) - 0.5)
    initial_b = -8
    cost = compute_cost(X_train, y_train, initial_w, initial_b, 0, False)

    msg('Cost at test w and b (non-zeros): {:.3f}'.format(cost))

    # computing gradient with random w and b
    dj_db, dj_dw  = compute_gradient(X_train, y_train, initial_w, initial_b, 0, False)
    msg(f'dj_db at test w and b: {dj_db}')
    msg(f'dj_dw at test w and b: {dj_dw.tolist()}')

    # gradient descent to find optimal w[] and b
    # settings for gradient descent
    iterations = 50000
    alpha = 0.001

    w, b, J_history, _ = gradient_descent(
        X_train ,y_train, initial_w, initial_b, 
        compute_cost, compute_gradient, alpha, iterations, 0, False
    )

    # plotting classification with learned w[] and b
    plot_decision_boundary(w, b, X_train, y_train)
    # Set the y-axis label
    plt.ylabel('Exam 2 score') 
    # Set the x-axis label
    plt.xlabel('Exam 1 score') 
    plt.legend(loc="upper right")
    plt.show()

    # Compute accuracy on our training set
    p = predict(X_train, w,b)
    msg('Train Accuracy: %f'%(np.mean(p == y_train) * 100))


def microprocessor_defect_model():
    print("*********** Microprocessors defect model ***********")
    # load dataset
    # feature x1 and x2 are the results for two differente Quality Assurance tests
    # the last column is 1 for OK, 0 for KO (defective chip)
    X_train, y_train = load_data(os.path.join("data", "microprocessor_test.csv"))

    # print X_train
    msg(f"X_train: {X_train[:5]}")
    msg(f"Type of X_train: {type(X_train)}")

    # print y_train
    msg(f"y_train: {y_train[:5]}")
    msg(f"Type of y_train: {type(y_train)}")

    msg('The shape of X_train is: ' + str(X_train.shape))
    msg('The shape of y_train is: ' + str(y_train.shape))
    msg('We have m = %d training examples' % (len(y_train)))

    # Plot examples
    plot_data(X_train, y_train[:], pos_label="Accepted", neg_label="Rejected")

    # Set the y-axis label
    plt.ylabel('Microchip Test 2') 
    # Set the x-axis label
    plt.xlabel('Microchip Test 1') 
    plt.legend(loc="upper right")
    plt.show()

    # feature mapping to have a non-linear classification boundary
    # add higher order polynomials as a combination of our two features x1 and x2
    msg(f"Original shape of data: {X_train.shape}")

    mapped_X = map_feature(X_train[:, 0], X_train[:, 1])
    msg(f"Shape after feature mapping: {mapped_X.shape}")

    msg(f"X_train[0]: {X_train[0]}")
    msg(f"mapped X_train[0]: {mapped_X[0]}")

    X_mapped = map_feature(X_train[:, 0], X_train[:, 1])
    np.random.seed(1)
    initial_w = np.random.rand(X_mapped.shape[1]) - 0.5
    initial_b = 0.5
    lambda_ = 0.5
    cost = compute_cost(X_mapped, y_train, initial_w, initial_b, lambda_, True)

    msg(f"Regularized cost: {cost}")

    # Initialize fitting parameters
    np.random.seed(1)
    initial_w = np.random.rand(X_mapped.shape[1])-0.5
    initial_b = 1.

    # Set regularization parameter lambda_ (you can try varying this)
    lambda_ = 0.01    

    # Some gradient descent settings
    iterations = 10000
    alpha = 0.01

    w, b, J_history, _ = gradient_descent(
        X_mapped, y_train, initial_w, initial_b,
        compute_cost, compute_gradient,
        alpha, iterations, lambda_, True
    )

    # classification boundary with learned w[] and b
    plot_decision_boundary(w, b, X_mapped, y_train)
    # Set the y-axis label
    plt.ylabel('Microchip Test 2') 
    # Set the x-axis label
    plt.xlabel('Microchip Test 1') 
    plt.legend(loc="upper right")
    plt.show()

    # Compute accuracy on the training set
    p = predict(X_mapped, w, b)

    msg('Train Accuracy: %f'%(np.mean(p == y_train) * 100))

if __name__ == "__main__":
    university_admission_model()
    microprocessor_defect_model()
