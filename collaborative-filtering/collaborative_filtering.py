import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

"""
|General Notation  | Description                                                           |
|:-----------------|:----------------------------------------------------------------------|
| r(i,j)           | scalar; = 1  if user j rated movie i  = 0  otherwise                  |
| y(i,j)           | scalar; = rating given by user j on movie i (if r(i,j) = 1 is defined)|
| w(j)             | vector; parameters for user j                                         |
| b(j)             | scalar; parameter for user j                                          |
| x(j)             | vector; feature ratings for movie i                                   |     
| n_u              | number of users num_users                                             |
| n_m              | number of movies num_movies                                           |
| n                | number of features num_features                                       |
| X                | matrix of vectors x(i)                                                |
| W                | matrix of vectors w(j)                                                |
| b                | vector of bias parameters b(j)                                        |
| R                | matrix of elements r(i,j)                                             |
"""

def normalize_ratings(Y, R):
    """
    Preprocess data by subtracting mean rating for every movie (every row).
    Only include real ratings R(i,j)=1.
    [Ynorm, Ymean] = normalize_ratings(Y, R) normalized Y so that each movie
    has a rating of 0 on average. Unrated moves then have a mean rating (0)
    Returns the mean rating in Ymean.
    """
    Ymean = (np.sum(Y*R,axis=1)/(np.sum(R, axis=1)+1e-12)).reshape(-1,1)
    Ynorm = Y - np.multiply(Ymean, R) 
    return(Ynorm, Ymean)

def load_precalc_params_small():
    file = open('./data/small_movies_X.csv', 'rb')
    X = np.loadtxt(file, delimiter = ",")

    file = open('./data/small_movies_W.csv', 'rb')
    W = np.loadtxt(file,delimiter = ",")

    file = open('./data/small_movies_b.csv', 'rb')
    b = np.loadtxt(file,delimiter = ",")
    b = b.reshape(1,-1)
    num_movies, num_features = X.shape
    num_users,_ = W.shape
    return(X, W, b, num_movies, num_features, num_users)

def load_ratings_small():
    file = open('./data/small_movies_Y.csv', 'rb')
    Y = np.loadtxt(file,delimiter = ",")

    file = open('./data/small_movies_R.csv', 'rb')
    R = np.loadtxt(file,delimiter = ",")
    return(Y,R)

def load_movie_list_pd():
    """ returns df with and index of movies in the order they are in in the Y matrix """
    df = pd.read_csv('./data/small_movie_list.csv', header=0, index_col=0,  delimiter=',', quotechar='"')
    mlist = df["title"].to_list()
    return(mlist, df)

# collaborative filtering cost function
def cost_function(X, W, b, Y, R, lambda_):
    """
    Returns the cost for the content-based filtering
    Args:
      X (ndarray (num_movies,num_features)): matrix of item features
      W (ndarray (num_users,num_features)) : matrix of user parameters
      b (ndarray (1, num_users)            : vector of user parameters
      Y (ndarray (num_movies,num_users)    : matrix of user ratings of movies
      R (ndarray (num_movies,num_users)    : matrix, where R(i, j) = 1 if the i-th movies was rated by the j-th user
      lambda_ (float): regularization parameter
    Returns:
      J (float) : Cost
    """
    nm, nu = Y.shape
    J = 0
    for i in range(nm): # for all movies
        for j in range(nu): # for all users
            if R[i,j] == 1: # if user j rated movie i
                J += ((W[j] @ X[i] + b[0][j]) - Y[i][j])**2

    regularization = (lambda_ / 2) * (np.sum(W**2) + np.sum(X**2))
    J = J * 1/2 + regularization
    return J

# collaborative filtering cost function (vectorized)
def cost_function_v(X, W, b, Y, R, lambda_):
    """
    Returns the cost for the content-based filtering
    Vectorized for speed. Uses tensorflow operations to be compatible with custom training loop.
    Args:
      X (ndarray (num_movies,num_features)): matrix of item features
      W (ndarray (num_users,num_features)) : matrix of user parameters
      b (ndarray (1, num_users)            : vector of user parameters
      Y (ndarray (num_movies,num_users)    : matrix of user ratings of movies
      R (ndarray (num_movies,num_users)    : matrix, where R(i, j) = 1 if the i-th movies was rated by the j-th user
      lambda_ (float): regularization parameter
    Returns:
      J (float) : Cost
    """
    j = (tf.linalg.matmul(X, tf.transpose(W)) + b - Y)*R
    J = 0.5 * tf.reduce_sum(j**2) + (lambda_/2) * (tf.reduce_sum(X**2) + tf.reduce_sum(W**2))
    return J

def main():
    # load initial values for X, W, b
    X, W, b, num_movies, num_features, num_users = load_precalc_params_small()

    # load ratings
    Y, R = load_ratings_small()

    print("Y", Y.shape, "R", R.shape)
    print("X", X.shape)
    print("W", W.shape)
    print("b", b.shape)
    print("num_features", num_features)
    print("num_movies",   num_movies)
    print("num_users",    num_users)

    tsmean =  np.mean(Y[0, R[0, :].astype(bool)])
    print(f"Average rating for movie 1 : {tsmean:0.3f} / 5")

    movie_list, movie_list_df = load_movie_list_pd()

    # Initialize my ratings to 0
    my_ratings = np.zeros(num_movies) 

    # Check the file small_movie_list.csv for id of each movie in our dataset
    # For example, Toy Story 3 (2010) has ID 2700, so to rate it "5"
    my_ratings[2700] = 5 

    # Persuasion (2007), rate it 2
    my_ratings[2609] = 2;

    # Set additional ratings
    my_ratings[929]  = 5   # Lord of the Rings: The Return of the King, The
    my_ratings[246]  = 5   # Shrek (2001)
    my_ratings[2716] = 3   # Inception
    my_ratings[1150] = 5   # Incredibles, The (2004)
    my_ratings[382]  = 2   # Amelie (Fabuleux destin d'Amélie Poulain, Le)
    my_ratings[366]  = 5   # Harry Potter and the Sorcerer's Stone (a.k.a. Harry Potter and the Philosopher's Stone) (2001)
    my_ratings[622]  = 5   # Harry Potter and the Chamber of Secrets (2002)
    my_ratings[988]  = 3   # Eternal Sunshine of the Spotless Mind (2004)
    my_ratings[2925] = 1   # Louis Theroux: Law & Disorder (2008)
    my_ratings[2937] = 1   # Nothing to Declare (Rien à déclarer)
    my_ratings[793]  = 5   # Pirates of the Caribbean: The Curse of the Black Pearl (2003)
    my_rated = [i for i in range(len(my_ratings)) if my_ratings[i] > 0]
    print(f"Len of my_rated = {len(my_rated)}")

    print('\nNew user ratings:\n')
    for i in range(len(my_ratings)):
        if my_ratings[i] > 0:
            print(f'Rated {my_ratings[i]} for  {movie_list_df.loc[i,"title"]}');

    # Reload ratings
    Y, R = load_ratings_small()

    # Add new user ratings to Y after the reload
    Y = np.c_[my_ratings, Y]

    # Add new user indicator matrix to R
    R = np.c_[(my_ratings != 0).astype(int), R]

    # Normalize the Dataset
    Ynorm, Ymean = normalize_ratings(Y, R)

    num_movies, num_users = Y.shape
    num_features = 100

    # Set Initial Parameters (W, X), use tf.Variable to track these variables
    tf.random.set_seed(1234) # for consistent results
    W = tf.Variable(tf.random.normal((num_users,  num_features),dtype=tf.float64),  name='W')
    X = tf.Variable(tf.random.normal((num_movies, num_features),dtype=tf.float64),  name='X')
    b = tf.Variable(tf.random.normal((1,          num_users),   dtype=tf.float64),  name='b')

    # Instantiate an optimizer
    optimizer = keras.optimizers.Adam(learning_rate=1e-1)

    iterations = 200
    lambda_ = 1
    for iter in range(iterations):
        # Use TensorFlow’s GradientTape
        # to record the operations used to compute the cost 
        with tf.GradientTape() as tape:
            # Compute the cost (forward pass included in cost)
            cost_value = cost_function_v(X, W, b, Ynorm, R, lambda_)

        # Use the gradient tape to automatically retrieve
        # the gradients of the trainable variables with respect to the loss
        grads = tape.gradient(cost_value, [X,W,b])

        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        optimizer.apply_gradients(zip(grads, [X,W,b]))

        # Log periodically to console
        if iter % 20 == 0:
            print(f"Training loss at iteration {iter}: {cost_value:0.1f}")


    # Make a prediction using trained weights and biases
    p = np.matmul(X.numpy(), np.transpose(W.numpy())) + b.numpy()
    print(f"shape of prediction matrix = {p.shape}")

    # restore the mean
    pm = p + Ymean
    
    my_predictions = pm[:,0] # predictions for user 0, the newly added user

    # sort predictions
    ix = tf.argsort(my_predictions, direction='DESCENDING')

    for i in range(17):
        j = ix[i]
        if j not in my_rated:
            print(f'Predicting rating {my_predictions[j]:0.2f} for movie {movie_list[j]}')

    print('\n\nOriginal vs Predicted ratings:\n')
    for i in range(len(my_ratings)):
        if my_ratings[i] > 0:
            print(f'Original {my_ratings[i]}, Predicted {my_predictions[i]:0.2f} for {movie_list[i]}')

    filter=(movie_list_df["number of ratings"] > 20)
    movie_list_df["pred"] = my_predictions
    movie_list_df = movie_list_df.reindex(columns=["pred", "mean rating", "number of ratings", "title"])
    movie_list_df.loc[ix[:300]].loc[filter].sort_values("mean rating", ascending=False)

if __name__ == '__main__':
    main()
