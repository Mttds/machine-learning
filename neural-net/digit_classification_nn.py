# 0 or 1 digit classification neural network
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

def load_data():
    X = np.load("data/X.npy")
    y = np.load("data/y.npy")
    X = X[0:1000]
    y = y[0:1000]
    return X, y

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

def show_data(X, y):
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)

    m, n = X.shape

    fig, axes = plt.subplots(8,8, figsize=(8,8))
    fig.tight_layout(pad=0.1)

    # display only 8x8 = 64 images out of training example of 1000
    for i,ax in enumerate(axes.flat):
        # Select random indices
        random_index = np.random.randint(m)
        
        # Select rows corresponding to the random indices and
        # reshape the image
        # display only elements corresponding to the row of the random index
        X_random_reshaped = X[random_index].reshape((20,20)).T
        
        # Display the image
        ax.imshow(X_random_reshaped, cmap='gray')
        
        # Display the label above the image
        ax.set_title(y[random_index,0])
        ax.set_axis_off()

    plt.show()

def show_rand_image(X, y, Yhat, misclassified):
    fig = plt.figure(figsize=(1, 1))
    errors = np.where(y != Yhat) if misclassified else np.where(y == Yhat)
    random_index = errors[0][0]
    X_random_reshaped = X[random_index].reshape((20, 20)).T
    plt.imshow(X_random_reshaped, cmap='gray')
    plt.title(f"{y[random_index,0]}, {Yhat[random_index, 0]}")
    plt.axis('off')
    plt.show()

def show_predicted_data(X, y, model):
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)

    m, n = X.shape

    fig, axes = plt.subplots(8,8, figsize=(8,8))
    fig.tight_layout(pad=0.1,rect=[0, 0.03, 1, 0.92]) #[left, bottom, right, top]

    for i,ax in enumerate(axes.flat):
        # Select random indices
        random_index = np.random.randint(m)
        
        # Select rows corresponding to the random indices and
        # reshape the image
        X_random_reshaped = X[random_index].reshape((20,20)).T
        
        # Display the image
        ax.imshow(X_random_reshaped, cmap='gray')
        
        # Predict using the Neural Network
        prediction = model.predict(X[random_index].reshape(1,400))
        if prediction >= 0.5:
            yhat = 1
        else:
            yhat = 0
        
        # Display the label above the image
        ax.set_title(f"{y[random_index,0]},{yhat}")
        ax.set_axis_off()
    fig.suptitle("Label, yhat", fontsize=16)
    plt.show()

def nn_model_tf(X):
    model = Sequential(
        [
            tf.keras.Input(shape=(X.shape[1],)), # specify input size (400 element vector, i.e 400 x features)
            Dense(units=25, activation='sigmoid'),
            Dense(units=15, activation='sigmoid'),
            #Dense(units=25, activation='relu'),
            #Dense(units=15, activation='relu'),
            Dense(units=1 , activation='sigmoid')
        ], name = "my_model" 
    )

    return model

def nn_model(x, W1, b1, W2, b2, W3, b3):
    #a1 = dense(x,  W1, b1, sigmoid)
    #a2 = dense(a1, W2, b2, sigmoid)
    #a3 = dense(a2, W3, b3, sigmoid)
    a1 = dense_vectorized(x,  W1, b1, sigmoid)
    a2 = dense_vectorized(a1, W2, b2, sigmoid)
    a3 = dense_vectorized(a2, W3, b3, sigmoid)
    return(a3)

def dense(a_in, W, b, g):
    """
    Computes dense layer
    Args:
      a_in (ndarray (n, )) : Data, 1 example 
      W    (ndarray (n,j)) : Weight matrix, n features per unit, j units
      b    (ndarray (j, )) : bias vector, j units  
      g    activation function (e.g. sigmoid, relu..)
    Returns
      a_out (ndarray (j,))  : j units
    """
    units = W.shape[1]
    a_out = np.zeros(units)
    for j in range(units):
        z = np.dot(a_in, W[:,j]) + b[j]
        a_out[j] = g(z)
    return(a_out)

def dense_vectorized(A_in, W, b, g):
    """
    Computes dense layer
    Args:
      A_in (ndarray (m,n)) : Data, m examples, n features each
      W    (ndarray (n,j)) : Weight matrix, n features per unit, j units
      b    (ndarray (1,j)) : bias vector, j units  
      g    activation function (e.g. sigmoid, relu..)
    Returns
      A_out (tf.Tensor or ndarray (m,j)) : m examples, j units
    """
    Z = np.matmul(A_in, W) + b
    A_out = g(Z)
    return(A_out)

def predict(p, y_target):
    if p >= 0.5:
        yhat = 1
    else:
        yhat = 0
    print("yhat = ", yhat, " label= ", y_target)

def main():
    # load dataset
    X, y = load_data()

    # explore data
    print ('The first element of X is: ', X[0]) # 400 elements (20x20 image)
    print ('The first element of y is: ', y[0,0]) # digit 0 or digit 1
    print ('The last element of y is: ', y[-1,0]) # digit 0 or digit 1

    print ('The shape of X is: ' + str(X.shape))
    print ('The shape of y is: ' + str(y.shape))

    show_data(X, y)

    model = nn_model_tf(X) # with tensorflow
    model.summary()
    [layer1, layer2, layer3] = model.layers

    W1,b1 = layer1.get_weights()
    W2,b2 = layer2.get_weights()
    W3,b3 = layer3.get_weights()
    print(f"W1 shape = {W1.shape}, b1 shape = {b1.shape}")
    print(f"W2 shape = {W2.shape}, b2 shape = {b2.shape}")
    print(f"W3 shape = {W3.shape}, b3 shape = {b3.shape}")

    for i in range(3):
        print(f"*** weights of layer {i+1} ***")
        print(model.layers[i].weights)

    # training model
    print("*********** TRAINING ***********")
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(0.001),
    )

    model.fit(
        X,y,
        epochs=20
    )

    prediction = model.predict(X[0].reshape(1,400)) # a zero in the training set
    print(f" predicting a zero: {prediction}")
    prediction = model.predict(X[500].reshape(1,400)) # a one in the training set
    print(f" predicting a one:  {prediction}")

    show_predicted_data(X, y, model)

    # copy trained weights and biases from tensorflow
    W1_tmp,b1_tmp = layer1.get_weights()
    W2_tmp,b2_tmp = layer2.get_weights()
    W3_tmp,b3_tmp = layer3.get_weights()

    # predict with nn forward prop not using tensorflow
    predict(nn_model(X[0], W1_tmp, b1_tmp, W2_tmp, b2_tmp, W3_tmp, b3_tmp), y[0][0])
    predict(nn_model(X[500], W1_tmp, b1_tmp, W2_tmp, b2_tmp, W3_tmp, b3_tmp), y[500][0])

    # get the full array of predictions (Yhat)
    P = nn_model(X, W1_tmp, b1_tmp, W2_tmp, b2_tmp, W3_tmp, b3_tmp)
    Yhat = (P >= 0.5).astype(int) # apply the threshold of 0.5 to every element in array P

    # good prediction
    show_rand_image(X, y, Yhat, False)

    # bad prediction
    show_rand_image(X, y, Yhat, True)

if __name__ == '__main__':
    main()
