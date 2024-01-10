from collections import defaultdict
import csv
from numpy import genfromtxt
import pickle as pickle
import numpy as np
import numpy.ma as ma
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import tabulate

"""
The movie content provided to the network is a combination of the original data and some 'engineered features'.
The original features are the year the movie was released and the movie's genre's presented as a one-hot vector.
There are 14 genres. The engineered feature is an average rating derived from the user ratings. 
The user content is composed of engineered features.
A per genre average rating is computed per user.
Additionally, a user id, rating count and rating average are available but not included in the training or prediction content.
They are carried with the data set because they are useful in interpreting data.
The training set consists of all the ratings made by the users in the data set.
Some ratings are repeated to boost the number of training examples of underrepresented genre's.
The training set is split into two arrays with the same number of entries, a user array and a movie/item array.  
"""

def load_data():
    ''' called to load preprepared data for the lab '''
    item_train = genfromtxt('./data/content_item_train.csv', delimiter=',')
    user_train = genfromtxt('./data/content_user_train.csv', delimiter=',')
    y_train    = genfromtxt('./data/content_y_train.csv', delimiter=',')
    with open('./data/content_item_train_header.txt', newline='') as f: # csv reader handles quoted strings better
        item_features = list(csv.reader(f))[0]
    with open('./data/content_user_train_header.txt', newline='') as f:
        user_features = list(csv.reader(f))[0]
    item_vecs = genfromtxt('./data/content_item_vecs.csv', delimiter=',')

    movie_dict = defaultdict(dict)
    count = 0

    with open('./data/content_movie_list.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for line in reader:
            if count == 0:
                count += 1  #skip header
                #print(line) print
            else:
                count += 1
                movie_id = int(line[0])
                movie_dict[movie_id]["title"] = line[1]
                movie_dict[movie_id]["genres"] = line[2]

    with open('./data/content_user_to_genre.pickle', 'rb') as f:
        user_to_genre = pickle.load(f)

    return(item_train, user_train, y_train, item_features, user_features, item_vecs, movie_dict, user_to_genre)

def pprint_train(x_train, features, vs, u_s, maxcount=5, user=True):
    """ Prints user_train or item_train nicely """
    if user:
        flist = [".0f", ".0f", ".1f",
                 ".1f", ".1f", ".1f", ".1f", ".1f", ".1f", ".1f", ".1f", ".1f", ".1f", ".1f", ".1f", ".1f", ".1f"]
    else:
        flist = [".0f", ".0f", ".1f", 
                 ".0f", ".0f", ".0f", ".0f", ".0f", ".0f", ".0f", ".0f", ".0f", ".0f", ".0f", ".0f", ".0f", ".0f"]

    head = features[:vs]
    if vs < u_s: print("error, vector start {vs} should be greater then user start {u_s}")
    for i in range(u_s):
        head[i] = "[" + head[i] + "]"
    genres = features[vs:]
    hdr = head + genres
    disp = [split_str(hdr, 5)]
    count = 0
    for i in range(0, x_train.shape[0]):
        if count == maxcount: break
        count += 1
        disp.append([x_train[i, 0].astype(int),
                     x_train[i, 1].astype(int),
                     x_train[i, 2].astype(float),
                     *x_train[i, 3:].astype(float)
                    ])
    table = tabulate.tabulate(disp, tablefmt='html', headers="firstrow", floatfmt=flist, numalign='center')
    return table

def split_str(ifeatures, smax):
    ''' split the feature name strings to tables fit '''
    ofeatures = []
    for s in ifeatures:
        if not ' ' in s: # skip string that already have a space
            if len(s) > smax:
                mid = int(len(s)/2)
                s = s[:mid] + " " + s[mid:]
        ofeatures.append(s)
    return ofeatures

def print_pred_movies(y_p, item, movie_dict, maxcount=10):
    """ print results of prediction of a new user. inputs are expected to be in
        sorted order, unscaled. """
    count = 0
    disp = [["y_p", "movie id", "rating ave", "title", "genres"]]

    for i in range(0, y_p.shape[0]):
        if count == maxcount:
            break
        count += 1
        movie_id = item[i, 0].astype(int)
        disp.append([np.around(y_p[i, 0], 1), item[i, 0].astype(int), np.around(item[i, 2].astype(float), 1),
                     movie_dict[movie_id]['title'], movie_dict[movie_id]['genres']])

    table = tabulate.tabulate(disp, tablefmt='html', headers="firstrow")
    return table

def gen_user_vecs(user_vec, num_items):
    """ given a user vector return:
        user predict maxtrix to match the size of item_vecs """
    user_vecs = np.tile(user_vec, (num_items, 1))
    return user_vecs

# predict on  everything, filter on print/use
def predict_uservec(user_vecs, item_vecs, model, u_s, i_s, scaler):
    """ given a scaled user vector, does the prediction on all movies in scaled print_item_vecs returns
        an array predictions sorted by predicted rating,
        arrays of user and item, sorted by predicted rating sorting index
    """
    y_p = model.predict([user_vecs[:, u_s:], item_vecs[:, i_s:]])
    y_pu = scaler.inverse_transform(y_p)

    if np.any(y_pu < 0):
        print("Error, expected all positive predictions")
    sorted_index = np.argsort(-y_pu, axis=0).reshape(-1).tolist()  #negate to get largest rating first
    sorted_ypu   = y_pu[sorted_index]
    sorted_items = item_vecs[sorted_index]
    sorted_user  = user_vecs[sorted_index]
    return(sorted_index, sorted_ypu, sorted_items, sorted_user)

def get_user_vecs(user_id, user_train, item_vecs, user_to_genre):
    """ given a user_id, return:
        user train/predict matrix to match the size of item_vecs
        y vector with ratings for all rated movies and 0 for others of size item_vecs """

    if not user_id in user_to_genre:
        print("error: unknown user id")
        return None
    else:
        user_vec_found = False
        for i in range(len(user_train)):
            if user_train[i, 0] == user_id:
                user_vec = user_train[i]
                user_vec_found = True
                break
        if not user_vec_found:
            print("error in get_user_vecs, did not find uid in user_train")
        num_items = len(item_vecs)
        user_vecs = np.tile(user_vec, (num_items, 1))

        y = np.zeros(num_items)
        for i in range(num_items):  # walk through movies in item_vecs and get the movies, see if user has rated them
            movie_id = item_vecs[i, 0]
            if movie_id in user_to_genre[user_id]['movies']:
                rating = user_to_genre[user_id]['movies'][movie_id]
            else:
                rating = 0
            y[i] = rating
    return(user_vecs, y)

def get_item_genres(item_gvec, genre_features):
    ''' takes in the item's genre vector and list of genre names
    returns the feature names where gvec was 1 '''
    offsets = np.nonzero(item_gvec)[0]
    genres = [genre_features[i] for i in offsets]
    return genres


def print_existing_user(y_p, y, user, items, ivs, uvs, movie_dict, maxcount=10):
    """ print results of prediction for a user who was in the database.
        Inputs are expected to be in sorted order, unscaled.
    """
    count = 0
    disp = [["y_p", "y", "user", "user genre ave", "movie rating ave", "movie id", "title", "genres"]]
    count = 0
    for i in range(0, y.shape[0]):
        if y[i, 0] != 0:  # zero means not rated
            if count == maxcount:
                break
            count += 1
            movie_id = items[i, 0].astype(int)

            offsets = np.nonzero(items[i, ivs:] == 1)[0]
            genre_ratings = user[i, uvs + offsets]
            disp.append([y_p[i, 0], y[i, 0],
                         user[i, 0].astype(int),      # userid
                         np.array2string(genre_ratings, 
                                         formatter={'float_kind':lambda x: "%.1f" % x},
                                         separator=',', suppress_small=True),
                         items[i, 2].astype(float),    # movie average rating
                         movie_id,
                         movie_dict[movie_id]['title'],
                         movie_dict[movie_id]['genres']])

    table = tabulate.tabulate(disp, tablefmt='html', headers="firstrow", floatfmt=[".1f", ".1f", ".0f", ".2f", ".1f"])
    return table

def nn_model(num_outputs, num_user_features, num_item_features):
    tf.random.set_seed(1)
    user_NN = tf.keras.models.Sequential([
        tf.keras.layers.Dense(units=256, activation='relu'),
        tf.keras.layers.Dense(units=128, activation='relu'),
        tf.keras.layers.Dense(units=num_outputs, activation='linear')
    ])

    item_NN = tf.keras.models.Sequential([
        tf.keras.layers.Dense(units=256, activation='relu'),
        tf.keras.layers.Dense(units=128, activation='relu'),
        tf.keras.layers.Dense(units=num_outputs, activation='linear')
    ])

    # create the user input and point to the base network
    input_user = tf.keras.layers.Input(shape=(num_user_features))
    vu = user_NN(input_user)
    vu = tf.linalg.l2_normalize(vu, axis=1)

    # create the item input and point to the base network
    input_item = tf.keras.layers.Input(shape=(num_item_features))
    vm = item_NN(input_item)
    vm = tf.linalg.l2_normalize(vm, axis=1)

    # compute the dot product of the two vectors vu and vm
    output = tf.keras.layers.Dot(axes=1)([vu, vm])

    # specify the inputs and output of the model
    model = tf.keras.Model([input_user, input_item], output)
    return model, user_NN, item_NN

def sq_dist(a,b):
    """
    Returns the squared distance between two vectors
    Args:
      a (ndarray (n,)): vector with n features
      b (ndarray (n,)): vector with n features
    Returns:
      d (float) : distance
    """
    d = np.sum((a - b)**2)   
    return d

def main():
    top10_df = pd.read_csv("./data/content_top10_df.csv")
    bygenre_df = pd.read_csv("./data/content_bygenre_df.csv")
    top10_df

    print(bygenre_df)

    # Load Data, set configuration variables
    item_train, user_train, y_train, item_features, user_features, item_vecs, movie_dict, user_to_genre = load_data()

    num_user_features = user_train.shape[1] - 3  # remove userid, rating count and ave rating during training
    num_item_features = item_train.shape[1] - 1  # remove movie id at train time
    uvs = 3  # user genre vector start
    ivs = 3  # item genre vector start
    u_s = 3  # start of columns to use in training, user
    i_s = 1  # start of columns to use in training, items
    print(f"Number of training vectors: {len(item_train)}")

    # movie rating given by each user
    print(f"y_train[:5]: {y_train[:5]}")

    # scale training data
    item_train_unscaled = item_train
    user_train_unscaled = user_train
    y_train_unscaled    = y_train

    scalerItem = StandardScaler()
    scalerItem.fit(item_train)
    item_train = scalerItem.transform(item_train)

    scalerUser = StandardScaler()
    scalerUser.fit(user_train)
    user_train = scalerUser.transform(user_train)

    scalerTarget = MinMaxScaler((-1, 1))
    scalerTarget.fit(y_train.reshape(-1, 1))
    y_train = scalerTarget.transform(y_train.reshape(-1, 1))

    print(np.allclose(item_train_unscaled, scalerItem.inverse_transform(item_train)))
    print(np.allclose(user_train_unscaled, scalerUser.inverse_transform(user_train)))

    item_train, item_test = train_test_split(item_train, train_size=0.80, shuffle=True, random_state=1)
    user_train, user_test = train_test_split(user_train, train_size=0.80, shuffle=True, random_state=1)
    y_train, y_test       = train_test_split(y_train,    train_size=0.80, shuffle=True, random_state=1)
    print(f"movie/item training data shape: {item_train.shape}")
    print(f"movie/item test data shape: {item_test.shape}")

    # build model
    num_outputs = 32 # output vector length for both the movie (items) and the user vector
    model, item_NN, user_NN = nn_model(num_outputs, num_user_features, num_item_features)
    model.summary()

    # train model
    tf.random.set_seed(1)
    cost_fn = tf.keras.losses.MeanSquaredError()
    opt = keras.optimizers.Adam(learning_rate=0.01)
    model.compile(optimizer=opt, loss=cost_fn)

    tf.random.set_seed(1)
    model.fit([user_train[:, u_s:], item_train[:, i_s:]], y_train, epochs=30)

    # evaluate the model on the test data set
    model.evaluate([user_test[:, u_s:], item_test[:, i_s:]], y_test)

    # new user
    new_user_id = 5000
    new_rating_ave = 0.0
    new_action = 0.0
    new_adventure = 5.0
    new_animation = 0.0
    new_childrens = 0.0
    new_comedy = 0.0
    new_crime = 0.0
    new_documentary = 0.0
    new_drama = 0.0
    new_fantasy = 5.0
    new_horror = 0.0
    new_mystery = 0.0
    new_romance = 0.0
    new_scifi = 0.0
    new_thriller = 0.0
    new_rating_count = 3

    user_vec = np.array([[new_user_id, new_rating_count, new_rating_ave,
                        new_action, new_adventure, new_animation, new_childrens,
                        new_comedy, new_crime, new_documentary,
                        new_drama, new_fantasy, new_horror, new_mystery,
                        new_romance, new_scifi, new_thriller]])

    # generate and replicate the user vector to match the number movies in the data set.
    user_vecs = gen_user_vecs(user_vec,len(item_vecs))

    # scale our user and item vectors
    suser_vecs = scalerUser.transform(user_vecs)
    sitem_vecs = scalerItem.transform(item_vecs)

    # make a prediction
    y_p = model.predict([suser_vecs[:, u_s:], sitem_vecs[:, i_s:]])

    # unscale y prediction 
    y_pu = scalerTarget.inverse_transform(y_p)

    # sort the results, highest prediction first
    sorted_index = np.argsort(-y_pu,axis=0).reshape(-1).tolist()  #negate to get largest rating first
    sorted_ypu   = y_pu[sorted_index]
    sorted_items = item_vecs[sorted_index]  #using unscaled vectors for display

    print(print_pred_movies(sorted_ypu, sorted_items, movie_dict, maxcount = 10))

if __name__ == '__main__':
    main()
