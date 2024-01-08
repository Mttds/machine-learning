# <span style="color:teal;font-weight:bold">machine learning</span>

## Requirements

conda: 4.10.3 (https://docs.conda.io/projects/miniconda/en/latest/)

Create env with: `conda create -n machine_learning python=3.10`

Requirements listed in requirements.txt

`pip list --format=freeze > requirements.txt`

```
pip install numpy
pip install matplotlib
pip install jupyterlab
pip install ipywidgets
pip install ipympl
pip install scikit-learn
pip install tensorflow
conda install -c anaconda sympy
pip install pandas
pip install xgboost
```

## Environment

Run the following command to activate the created anaconda python virtual environment.

`source conda-env.sh`

## Jupyter notebook

After having sourced conda-env.sh:

Run `./run_jupyterlab.sh` to start the Jupyter lab server.

## Content

### Supervised Learning

#### <span style="color:teal;font-weight:bold">Linear Regression</span><hr/>

_Folder_:

- linear-regression

_Python_:

- model_representation.py: univariate linear regression example
- vectorization.py: applying vectorization to loops and linear algebra operations with Numpy
- linear_regression.py: example of linear regression model
- locally_weighted_linear_regression.py: locally weighted version of linear regression for sample data that does not fit well a linear regression using both batch gradient descent and stochastic gradient descent to find the optimal w (theta) parameter.

_C_:

- gradient_descent.c: gradient descent implementation for univariate linear regression (one x1 feature) using OpenMP. The library is compiled as a shared object (.so) and called from Python in linear_regression.py

_Jupyter Notebook_:

- cost_function.ipynb: python notebook for cost function visualization used in our univariate linear regression model
- gradient_descent.ipynb: python notebook for gradient descent implementation and visualization for the optimal w,b parameters of the univariate
- multiple_linear_regression.ipynb: python notebook for multiple univariate linear regression model
- feature_scaling_and_learning_rt.ipynb: feature scaling with various methods and choosing the correct learning rate for the model
- feature_eng_and_polynomial_regression.ipynb: adding new features based on the existing features (feature engineering) and modeling non-linear functions

_Useful links_:

- Least Squares (error/cost function) and Normal Equations vs Gradient Descent: https://math.mit.edu/icg/resources/teaching/18.085-spring2015/LeastSquares.pdf

#### <span style="color:teal;font-weight:bold">Classification</span><hr/>

_Folder_:

- classification

_Python_:

- logistic_regression.py: implementation of two prediction models using logistic regression (classification)

_Jupyter Notebook_:

- classification.ipynb: classification with linear regression attempt and with logistic regression
- cost_function.ipynb: cost function for logistic regression and logistic loss
- gradient_descent.ipynb: gradient descent for the logistic cost function
- regularization.ipynb: regularization technique to combat under and over fitting

Correct for over/under fitting:

- Collect more training data for the model
- Regularize weights for features
- Add/remove features (feature engineering)

#### <span style="color:teal;font-weight:bold">Neural Networks</span><hr/>

_Folder_:

- neural-net

_Python_:

- digit_classification_nn.py: classifies digit 0 or digit 1 using a neural network (both with TensorFlow and manual implementation [using tf trained weights and biases])

- multiclass_digit_classification_nn.py: classifies digits 0 through 9 using a neural network with softmax.

_Jupyter Notebook_:

- forward_prop_nn_tensorflow.ipynb: forward propagation simple neural network using TensorFlow
- forward_prop_nn_numpy.ipynb: forward propagation simple neural network without using TensorFlow
- multiclass_nn_softmax.ipynb: softmax function for the output layer of a neural net for multiclass classification problems

To fix a high bias problem:

- adding polynomial features
- getting additional features
- decreasing the regularization parameter

To fix a high variance problem:

- increasing the regularization parameter
- smaller sets of features
- more training examples

#### <span style="color:teal;font-weight:bold">Decision Trees</span><hr/>

_Folder_:

- decision-tree

_Python_:

- decision_tree.py: implementation of a decision tree algorithm with no libraries

_Jupyter Notebook_:

- decision_trees.ipynb: example decision tree (and one-hot encoding)
- ensembles.ipynb: XGBoost, Random Forest examples (bagging, boosting decision trees)

### Unsupervised Learning

#### <span style="color:teal;font-weight:bold">K-means</span><hr/>

_Folder_:

- kmeans

_Jupyter Notebook_:

- image_compression.ipynb: example of k-means application

#### <span style="color:teal;font-weight:bold">Anomaly Detection</span><hr/>

_Folder_:

- anomaly-detection

_Jupyter Notebook_:

- anomaly_detection.ipynb: example of anomlay detection application

### Reinforcement Learning
