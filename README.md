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

### Linear Regression

_Folder_:

- linear-regression

_Python_:

- model_representation.py: univariate linear regression example

_Jupyter Notebook_:

- cost_function.ipynb: python notebook for cost function visualization used in our univariate linear regression model
