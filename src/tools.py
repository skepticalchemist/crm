import matplotlib as plt
import pandas as pd
import numpy as np
import sklearn.metrics as metrics
from sklearn.model_selection import learning_curve
from sklearn.tree import DecisionTreeRegressor



def plot_learning_curve(X, y, maxdepth, estimator, plt):
    """
    This function plot the learning curve for a model.
    
    ...
    
    Attributes
    ----------
    X : 
         feature matrix
    y : 
         target variable vector
    maxdepth: int
         max_depth parameter for classifier
    """
    # create cv training and test scores for various training set sizes
    train_sizes, train_scores, test_scores = learning_curve(estimator,
                                                            X,                                     # feature matrix
                                                            y,                                     # target vector
                                                            cv=10,                                 # number of folds in cross-validation
                                                            scoring='neg_mean_squared_error',      # metric
                                                            n_jobs=-1,                             # use all computer cores,
                                                            train_sizes=np.linspace(0.01, 1.0, 30) # 30 different sizes of the training set
                                                            )
    # create means and standart deviations of training set scores
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)

    # create means and standart deviations of test set scores
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # draw lines
    plt.plot(train_sizes, train_mean, '--', color='#111111', label="Training score")
    plt.plot(train_sizes, test_mean, color='#111111', label="Cross-validation score")

    # draw bands
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="#DDDDDD")
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="#f4d0d7")
       
    # create plot    
    plt.title("Learning curve")
    plt.xlabel("Training set size", fontsize=18)
    plt.ylabel("mse", fontsize=18)
    plt.legend(loc="best")
    plt.tight_layout()



def plot_feature_importance_tree(model, dataset, title, plt):
    """
    This function calculates the feature importance and plot a horizontal bar chart.
    ...
    Attributes
    ----------
    model : sklearn.tree._classes.DecisionTreeRegressor
         the regression decision tree model
    dataset : pandas.core.frame.DataFrame
         the training dataset
    title: str
         the title for the plot
    """
    n_features = dataset.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), dataset.columns)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.title(title)