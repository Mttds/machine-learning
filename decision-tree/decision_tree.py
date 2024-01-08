"""
- `X_train` contains three features for each example 
    - Brown Color (a value of `1` indicates "Brown" cap color and `0` indicates "Red" cap color)
    - Tapering Shape (a value of `1` indicates "Tapering Stalk Shape" and `0` indicates "Enlarging" stalk shape)
    - Solitary  (a value of `1` indicates "Yes" and `0` indicates "No")

- `y_train` is whether the mushroom is edible 
    - `y = 1` indicates edible
    - `y = 0` indicates poisonous
"""

import numpy as np
import matplotlib.pyplot as plt


def load_data():
    """
    Generate sample data
    """
    X_train = np.array([[1,1,1],[1,0,1],[1,0,0],[1,0,0],[1,1,1],[0,1,1],[0,0,0],[1,0,1],[0,1,0],[1,0,0]])
    y_train = np.array([1,1,0,0,1,0,0,1,1,0]) # 1 for edible, 0 for poisonous

    return X_train, y_train

def compute_entropy(y):
    """
    Computes the entropy for 

    Args:
       y (ndarray): Numpy array indicating whether each example at a node is
           edible (`1`) or poisonous (`0`)

    Returns:
        entropy (float): Entropy at that node

    """
    entropy = 0.
    p = sum([i for i in y if i == 1]) / len(y) if len(y) > 0 else 0
    entropy = 0 if p == 1 or p == 0 else -p * np.log2(p) - (1-p) * np.log2(1-p)    
    return entropy

def split_tree(X, node_indices, feature):
    """
    Splits the data at the given node into
    left and right branches

    Args:
        X (ndarray):             Data matrix of shape(n_samples, n_features)
        node_indices (list):     List containing the active indices. I.e, the samples being considered at this step.
        feature (int):           Index of feature to split on

    Returns:
        left_indices (list):     Indices with feature value == 1
        right_indices (list):    Indices with feature value == 0
    """
    left_indices = []
    right_indices = []
    
    for i in node_indices:
        left_indices.append(i) if X[i][feature] == 1 else right_indices.append(i)

    return left_indices, right_indices

def compute_information_gain(X, y, node_indices, feature):
    """
    Compute the information of splitting the node on a given feature

    Args:
        X (ndarray):            Data matrix of shape(n_samples, n_features)
        y (array like):         list or ndarray with n_samples containing the target variable
        node_indices (ndarray): List containing the active indices. I.e, the samples being considered in this step.
   
    Returns:
        cost (float):        Cost computed

    """
    left_indices, right_indices = split_tree(X, node_indices, feature)

    X_node, y_node = X[node_indices], y[node_indices]
    X_left, y_left = X[left_indices], y[left_indices]
    X_right, y_right = X[right_indices], y[right_indices]

    w_left = len(X_left) / len(X_node)
    w_right = len(X_right) / len(X_node)
    information_gain = compute_entropy(y_node) - (w_left * compute_entropy(y_left) + w_right * compute_entropy(y_right))

    return information_gain

def get_best_split(X, y, node_indices):   
    """
    Returns the optimal feature and threshold value
    to split the node data 
    
    Args:
        X (ndarray):            Data matrix of shape(n_samples, n_features)
        y (array like):         list or ndarray with n_samples containing the target variable
        node_indices (ndarray): List containing the active indices. I.e, the samples being considered in this step.

    Returns:
        best_feature (int):     The index of the best feature to split
    """
    num_features = X.shape[1]
    best_feature = -1
    best_gain = 0
    for i in range(num_features):
        info_gain = compute_information_gain(X, y, node_indices, feature=i)
        if info_gain > best_gain:
            best_feature = i
            best_gain = info_gain

    return best_feature

def build_tree_recursive(X, y, node_indices, branch_name, tree, max_depth, current_depth):
    """
    Build a tree using the recursive algorithm that split the dataset into 2 subgroups at each node.
    This function just prints the tree.

    Args:
        X (ndarray):            Data matrix of shape(n_samples, n_features)
        y (array like):         list or ndarray with n_samples containing the target variable
        node_indices (ndarray): List containing the active indices. I.e, the samples being considered in this step.
        branch_name (string):   Name of the branch. ['Root', 'Left', 'Right']
        max_depth (int):        Max depth of the resulting tree. 
        current_depth (int):    Current depth. Parameter used during recursive call.
   
    """ 

    # maximum depth reached - stop splitting
    if current_depth == max_depth:
        formatting = " "*current_depth + "-"*current_depth
        print(formatting, "%s leaf node with indices" % branch_name, node_indices)
        return

    # otherwise, get best split and split the data
    # get the best feature and threshold at this node
    best_feature = get_best_split(X, y, node_indices) 
    formatting = "-"*current_depth
    print("%s Depth %d, %s: Split on feature: %d" % (formatting, current_depth, branch_name, best_feature))

    # split the dataset at the best feature
    left_indices, right_indices = split_tree(X, node_indices, best_feature)
    tree.append((left_indices, right_indices, best_feature))

    # continue splitting the left and the right child. Increment current depth
    build_tree_recursive(X, y, left_indices, "Left", tree, max_depth, current_depth+1)
    build_tree_recursive(X, y, right_indices, "Right", tree, max_depth, current_depth+1)

def main():
    X_train, y_train = load_data()
    print('The shape of X_train is:', X_train.shape)
    print('The shape of y_train is: ', y_train.shape)
    print('Number of training examples (m):', len(X_train))

    root_indices = [i for i in range(len(X_train))]
    best_feature = get_best_split(X_train, y_train, root_indices)
    print("Best feature to split on: %d" % best_feature)
    build_tree_recursive(X_train, y_train, root_indices, "Root", [], max_depth=2, current_depth=0)

if __name__ == '__main__':
    main()
