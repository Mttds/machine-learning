import numpy as np
import matplotlib.pyplot as plt
import random

# Batch Gradient Descent
def lwlr_batch_gradient_descent(x, y, query_point, tau, alpha, num_iterations):
    m = len(x)
    x = np.c_[np.ones(m), x] # add a column of ones for the intercept term
    y = y.reshape(-1, 1) # reshape y array (i.e. row vector) into a m x 1 matrix (i.e column vector)
    weights = np.zeros((m, m)) # diagolal matrix to hold weights

    for i in range(m):
        weights[i][i] = np.exp(-(x[i][1] - query_point) ** 2 / (2 * tau ** 2))

    theta = np.zeros((2, 1)) # init theta(1) and theta(0) (i.e. bias, intercept term) to 0)

    for _ in range(num_iterations):
        hypothesis = np.dot(x, theta)
        loss = hypothesis - y
        cost = np.sum(weights * loss ** 2) / (2 * m) # weighted cost function
        gradient = np.dot(x.T, weights.dot(loss)) / m
        theta -= alpha * gradient # update theta for this iteration

    return np.dot(np.array([1, query_point]), theta)[0], theta

# Stochastic Gradient Descent
def lwlr_stochastic_gradient_descent(x, y, query_point, tau, alpha, num_iterations):
    m = len(x)
    x = np.c_[np.ones(m), x] # add a column of ones for the intercept term
    y = y.reshape(-1, 1) # reshape y array (i.e. row vector) into a m x 1 matrix (i.e column vector)
    weights = np.zeros((m, m)) # diagonal matrix to hold weights

    for i in range(m):
        weights[i][i] = np.exp(-(x[i][1] - query_point) ** 2 / (2 * tau ** 2))

    theta = np.zeros((2, 1)) # init theta(1) and theta(0) (i.e. bias, intercept term) to 0)

    for _ in range(num_iterations):
        i = random.randint(0, m-1) # random example for sgd
        hypothesis = np.dot(x[i], theta)
        loss = hypothesis - y[i]
        gradient = x[i].reshape(-1, 1) * weights[i][i] * loss
        theta -= alpha * gradient # update theta for this iteration

    return np.dot(np.array([1, query_point]), theta)[0], theta

# sample data
def main():
    np.random.seed(42)
    x = np.random.uniform(low=0, high=30, size=200)
    y = 10 * np.sin(0.5 * x) + np.cos(3 * x) + 2 * x
    plt.figure(figsize = (12, 8))
    plt.scatter(x, y, label='Sample Data')
    plt.ylabel('y')
    plt.xlabel('x')
    plt.show()

    print(f"Shape of x = {x.shape}")
    print(f"Num elements in training set: x = {len(x)}, y = {len(y)}")

    # query point for prediction (i.e locally weighted around this point)
    query_point = 15
    tau = 5 # bandwidth
    alpha = 0.001 # learning rate
    num_iterations = 10000

    # predictions
    p, theta_bgd = lwlr_batch_gradient_descent(x, y, query_point, tau, alpha, num_iterations)
    print(f"Prediction using Batch Gradient Descent LWLR: {p}")
    p, theta_sgd = lwlr_stochastic_gradient_descent(x, y, query_point, tau, alpha, num_iterations)
    print(f"Prediction using Stochastic Gradient Descent LWLR: {p}")

    # plot data points
    plt.figure(figsize=(12, 8))
    plt.scatter(x, y, label='Sample Data')

    # linear regression with feature weight (theta) found by batch gradient descent
    x_vals = np.linspace(min(x), max(x), 100)
    x_batch = np.c_[np.ones(100), x_vals]
    y_batch = x_batch.dot(theta_bgd)
    plt.plot(x_vals, y_batch, color='red', label='Fitted Line (Batch GD LWLR)')

    # linear regression with feature weight (theta) found by stochastic gradient descent
    y_stochastic = x_batch.dot(theta_sgd)
    plt.plot(x_vals, y_stochastic, color='green', label='Fitted Line (Stochastic GD LWLR)')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Locally Weighted Linear Regression')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
