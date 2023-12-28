"""
The data set contains 5000 training examples of handwritten digits.  

- Each training example is a 20-pixel x 20-pixel grayscale image of the digit. 
- Each pixel is represented by a floating-point number indicating the grayscale intensity at that location. 
- The 20 by 20 grid of pixels is “unrolled” into a 400-dimensional vector. 
- Each training examples becomes a single row in our data matrix `X`. 
- This gives us a 5000 x 400 matrix `X` where every row is a training example of a handwritten digit image.

The second part of the training set is a 5000 x 1 dimensional vector `y` that contains labels for the training set
- `y = 0` if the image is of the digit `0`, `y = 4` if the image is of the digit `4` and so on.

(Subset of MNIST dataset)
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import linear, relu, sigmoid
import matplotlib.pyplot as plt
plt.style.use('./ml.mplstyle')

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

from utils_common import *
np.set_printoptions(precision=2)

def widgvis(fig):
    fig.canvas.toolbar_visible = False
    fig.canvas.header_visible = False
    fig.canvas.footer_visible = False

def load_data():
    X = np.load("data/X.npy")
    y = np.load("data/y.npy")
    return X, y

def my_softmax(z):  
    """ Softmax converts a vector of values to a probability distribution.
    Args:
      z (ndarray (N,))  : input data, N features
    Returns:
      a (ndarray (N,))  : softmax of z
    """
    N = z.shape[0]
    a = np.zeros(N) # init a
    
    # for each class/category
    for j in range(N):
        a[j] = np.exp(z[j]) / sum(np.exp(z[k]) for k in range(N))
    return a

def show_data(X, y):
    # select 64 rows of our m element X matrix (5000 examples)
    # and show the images of 20x20 grayscalevalues
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)

    m, n = X.shape

    fig, axes = plt.subplots(8,8, figsize=(5,5))
    fig.tight_layout(pad=0.13,rect=[0, 0.03, 1, 0.91]) #[left, bottom, right, top]

    widgvis(fig)
    for i,ax in enumerate(axes.flat):
        # Select random indices
        random_index = np.random.randint(m)
        
        # Select rows corresponding to the random indices and
        # reshape the image
        X_random_reshaped = X[random_index].reshape((20,20)).T
        
        # Display the image
        ax.imshow(X_random_reshaped, cmap='gray')
        
        # Display the label above the image
        ax.set_title(y[random_index,0])
        ax.set_axis_off()
        fig.suptitle("Label, image", fontsize=14)
    plt.show()

def show_predictions(X, y, model):
    # select 64 rows of our m element X matrix (5000 examples)
    # and show the images of 20x20 grayscalevalues
    # along with the predicted digit by the model
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)

    m, n = X.shape

    fig, axes = plt.subplots(8,8, figsize=(5,5))
    fig.tight_layout(pad=0.13,rect=[0, 0.03, 1, 0.91]) #[left, bottom, right, top]
    widgvis(fig)
    for i,ax in enumerate(axes.flat):
        # Select random indices
        random_index = np.random.randint(m)
        
        # Select rows corresponding to the random indices and
        # reshape the image
        X_random_reshaped = X[random_index].reshape((20,20)).T
        
        # Display the image
        ax.imshow(X_random_reshaped, cmap='gray')
        
        # Predict using the Neural Network
        p = model.predict(X[random_index].reshape(1,400))
        p = tf.nn.softmax(p)
        yhat = np.argmax(p)
        
        # Display the label above the image
        ax.set_title(f"{y[random_index,0]},{yhat}",fontsize=10)
        ax.set_axis_off()
    fig.suptitle("Label, yhat", fontsize=14)
    plt.show()

def neural_network(X):
    tf.random.set_seed(1234) # for consistent results
    model = Sequential(
        [               
            tf.keras.Input(shape=(X.shape[1],)), # inputs 20x20 grayscale values
            Dense(units=25, activation='relu'),
            Dense(units=15, activation='relu'),
            Dense(units=10, activation='linear') # use from_logits=True in SparseCategoricalCrossentropy and then apply softmax
        ], name = "my_model" 
    )

    return model

def plot_loss_tf(history):
    fig,ax = plt.subplots(1,1, figsize = (4,3))
    widgvis(fig)
    ax.plot(history.history['loss'], label='loss')
    ax.set_ylim([0, 2])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('loss (cost)')
    ax.legend()
    ax.grid(True)
    plt.show()


def display_image(X):
    """ display a single digit. The input is one digit (400,). """
    fig, ax = plt.subplots(1,1, figsize=(0.5,0.5))
    widgvis(fig)
    X_reshaped = X.reshape((20,20)).T
    # Display the image
    ax.imshow(X_reshaped, cmap='gray')
    plt.show()

def display_errors(model,X,y):
    f = model.predict(X)
    yhat = np.argmax(f, axis=1)
    idxs = np.where(yhat != y[:,0])[0]
    if len(idxs) == 0:
        print("no errors found")
    else:
        cnt = min(8, len(idxs))
        fig, ax = plt.subplots(1,cnt, figsize=(5,1.2))
        fig.tight_layout(pad=0.13,rect=[0, 0.03, 1, 0.80]) #[left, bottom, right, top]
        widgvis(fig)

        for i in range(cnt):
            j = idxs[i]
            X_reshaped = X[j].reshape((20,20)).T

            # Display the image
            ax[i].imshow(X_reshaped, cmap='gray')

            # Predict using the Neural Network
            prediction = model.predict(X[j].reshape(1,400))
            prediction_p = tf.nn.softmax(prediction)
            yhat = np.argmax(prediction_p)

            # Display the label above the image
            ax[i].set_title(f"{y[j,0]},{yhat}",fontsize=10)
            ax[i].set_axis_off()
            fig.suptitle("Label, yhat", fontsize=12)
        plt.show()
    return(len(idxs))

def main():
    # load dataset
    X, y = load_data()

    print ('The first element of X is: ', X[0])
    print ('The first element of y is: ', y[0,0])
    print ('The last element of y is: ', y[-1,0])
    print ('The shape of X is: ' + str(X.shape))
    print ('The shape of y is: ' + str(y.shape))

    show_data(X, y)

    # create the neural network
    model = neural_network(X)
    model.summary()

    [layer1, layer2, layer3] = model.layers
    W1,b1 = layer1.get_weights()
    W2,b2 = layer2.get_weights()
    W3,b3 = layer3.get_weights()
    print(f"W1 shape = {W1.shape}, b1 shape = {b1.shape}")
    print(f"W2 shape = {W2.shape}, b2 shape = {b2.shape}")
    print(f"W3 shape = {W3.shape}, b3 shape = {b3.shape}")

    # run backpropagation to minimize loss given by W[] and b parameters at each layer
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    )
    history = model.fit(X, y, epochs=40)

    # history of total loss (cost) as a function of the epochs (iteration of gradient descent in backpropagation)
    plot_loss_tf(history)

    # make a prediction with the trained model
    image_of_two = X[1015] # 1016th training row
    display_image(image_of_two)

    p = model.predict(image_of_two.reshape(1,400)) # prediction

    print(f"predicting a Two: \n{p}")
    print(f"Largest Prediction index: {np.argmax(p)}")

    # with softmax to get a probability value between 0 and 1
    p = tf.nn.softmax(p)

    print(f" predicting a Two. Probability vector: \n{p}")
    print(f"Total of predictions: {np.sum(p):0.3f}")
    yhat = np.argmax(p)

    print(f"np.argmax(p): {yhat}")
    show_predictions(X, y, model)

    # display error rate out of the training sample
    print(f"{display_errors(model,X,y)} errors out of {len(X)} images")

if __name__ == '__main__':
    main()
