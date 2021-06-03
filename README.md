# Optimizing prescription rate


# Solution
Initially, an initial exploratory data analysis, data cleaning, and feature engineering were performed. Then regression models using linear regression, decision tree, random forest and xgboost were built and evaluated. The customer segmentation was addressed using k-means clustering.

# Setup

## Setup a conda environment

Create and activate a conda environment of your choice, for example, 'crm':

```python
conda create --name crm python==3.8
conda activate crm
```

## Install the packages through the command

```python
pip install -r requirements.txt
```

# Execution sequence

The notebooks should be executed in the following order:

* _1_EDA.ipynb_: 
   - performs an exploratory data analysis and some data cleaning   
* _2_Feature_engineering.ipynb_:
   - performs feature engineering and additional data cleaning when needed
* _3_Modeleling.ipynb_:
   - build, select, compare and evaluate regression models
* _4_Customer_segmentation.ipynb_:
   - performs clustering by k-means


# Directory structure or the repository
```{bash}
.
├── assets
├── data
├── notebook
├── reports
└── src

```
