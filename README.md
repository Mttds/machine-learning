# machine learning

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
```

## Environment

Run the following command to activate the created anaconda python virtual environment.

`source conda-env.sh`

## Jupyter notebook

After having sourced conda-env.sh:

Run `./run_jupyterlab.sh` to start the Jupyter lab server.

## Content

### Supervised Learning

#### Linear Regression

_Folder_:

- linear-regression

_Python_:

- model_representation.py: univariate linear regression example
- vectorization.py: applying vectorization to loops and linear algebra operations with Numpy

_Jupyter Notebook_:

- cost_function.ipynb: python notebook for cost function visualization used in our univariate linear regression model
- gradient_descent.ipynb: python notebook for gradient descent implementation and visualization for the optimal w,b parameters of the univariate
- multiple_linear_regression.ipynb: python notebook for multiple univariate linear regression model
- feature_scaling_and_learning_rt.ipynb: feature scaling with various methods and choosing the correct learning rate for the model
- feature_eng_and_polynomial_regression.ipynb: adding new features based on the existing features (feature engineering) and modeling non-linear functions

_Useful links_:

- Least Squares (error/cost function) and Normal Equations vs Gradient Descent: https://math.mit.edu/icg/resources/teaching/18.085-spring2015/LeastSquares.pdf

#### Classification

_Folder_:

- classification

_Jupyter Notebook_:

- classification.ipynb: classification with linear regression attempt and with logistic regression
