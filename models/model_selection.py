"""
Module for the selection of machine learning models.

There are several different functions which can perform the model selection: all of them have an intuitive interface, but
are also powerful and flexible.
In addition, almost all these functions can optionally make plots, which sum up the performed selection in a visual way.

These different functions perform the model selection in different contexts, i.e. each function is specifically meant for a
specific scenario. Certain contexts are more specific, and other are more general.
On the whole, there are six different model selection functions, divided into two main groups:
    1. functions that perform the model selection with respect to a **single dataset**;
    2. functions that perform the model selection with respect to **multiple datasets**.

The six functions, sorted from the most specific context to the most general one, are:
    - *hyperparameter_validation*, *hyperparameters_validation*, *models_validation* (single dataset);
    - *datasets_hyperparameter_validation*, *datasets_hyperparameters_validation*, *datasets_models_validation* (multiple
      datasets).

This module deeply uses the **numpy** library. It is built on the top of it. In fact, the datasets are represented as np.array.
Moreover, the plots are made using the **matplotlib** library. In addition, it is built on the top of the **sklearn** module:
- the machine learning models are represented as sklearn models (i.e. sklearn estimators);
- under the hood, the selection is performed using the grid search cross validation provided by sklearn (i.e.
GridSearchCV);
- several other operations are done using the functionalities provided by sklearn.

This module, besides the model selection functions, contains also some utilities:
- the PolynomialRegression class;
- some utility functions.

"""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.utils import resample
from sklearn.model_selection import learning_curve, train_test_split, cross_val_score, TimeSeriesSplit, GridSearchCV
from sklearn.metrics import auc, classification_report, mean_squared_error, accuracy_score, roc_curve, ConfusionMatrixDisplay
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score, mean_absolute_error, roc_auc_score, f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
# import precision_recall_curve from sklearn
from sklearn.metrics import precision_recall_curve, average_precision_score

import logging
# logging.basicConfig(filename='modeling.log', filemode='a', format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO)
# logger = logging.getLogger()
# handler = logging.FileHandler('modeling.log', mode='a')
# formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
# handler.setFormatter(formatter)
# logger.addHandler(handler)
# logger.setLevel(logging.INFO)

# logging.info('Started logging')




#----------------------------------------------------------------------------------------------------------------------------
# POLYNOMIAL REGRESSOR MODEL

class PolynomialRegression(BaseEstimator):
    """
    Polynomial regression model.

    It's a sklearn model: it's compliant to the sklearn estimators interface.
    `Example <https://scikit-learn.org/stable/developers/develop.html>`_

    Parameters
    ----------
    degree: int
        Degree to apply for the polynomial transformation.

    Notes
    ----------
    The polynomial transformation is performed using the sklearn PolynomialFeatures.
    """

    def __init__(self, degree=1):
        self.degree=degree

    def fit(self, X, y):
        self.poly_transformer = PolynomialFeatures(self.degree, include_bias=False)
        self.poly_transformer.fit(X)
        X = self.poly_transformer.transform(X)
        self.model = LinearRegression(fit_intercept=True)
        self.model.fit(X,y)
        return self

    def predict(self, X):
        X = self.poly_transformer.transform(X)
        return self.model.predict(X)

    def get_params(self, deep=True):
        return {"degree": self.degree}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self




#----------------------------------------------------------------------------------------------------------------------------
# UTILITY FUNCTIONS


def compute_train_val_test(X, y, model, scale=False, test_size=0.2, time_series=False, random_state=123, n_folds=5,
                           regr=True, **kwargs):
    """
    Compute the training-validation-test scores for the given model on the given dataset.

    The training and test scores are simply computed by splitting the dataset into the training and test sets. The validation
    score is performed applying the cross validation on the training set.

    Parameters
    ----------
    X: np.array
        Two-dimensional np.array, containing the explanatory features of the dataset.
    y: np.array
        Mono dimensional np.array, containing the response feature of the dataset.
    model: sklearn.base.BaseEstimator
        Model to evaluate.
    scale: bool
        Indicates whether to scale or not the features in `X`.
        (The scaling is performed using the sklearn MinMaxScaler).
    test_size: float
        Decimal number between 0 and 1, which indicates the proportion of the test set.
    time_series: bool
        Indicates if the given dataset is a time series dataset (i.e. datasets indexed by days).
        (This affects the computing of the scores).
    random_state: int
        Used in the training-test splitting of the dataset.
    n_folds: int
        Indicates how many folds are made in order to compute the k-fold cross validation.
        (It's used only if `time_series` is False).
    regr: bool
        Indicates if it's either a regression or a classification problem.

    Returns
    ----------
    train_score: float
    val_score: float
    test_score: float

    Notes
    ----------
    - If `regr` is True, the returned scores are errors, computed using the MSE formula (i.e. Mean Squared Error).
      Otherwise, the returned scores are accuracy measures.
    - If `time_series` is False, the training-test splitting of the dataset is made randomly. In addition, the cross
      validation strategy performed is the classic k-fold cross validation: the number of folds is specified by `n_folds`.
      Otherwise, if `time_series` is True, the training-test sets are obtained simply by splitting the dataset into two
      contiguous parts. In addition, the cross validation strategy performed is the sklearn TimeSeriesSplit.
    """

    # if regr:
    #     scoring="neg_mean_squared_error"
    # else:
    #     scoring="accuracy"

    # Split into training e test.
    if not time_series : # Random splitting (not time series)
        X_train_80, X_test, y_train_80, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    else: # time series splitting
        train_len = int(X.shape[0]*(1-test_size))
        X_train_80 = X[:train_len]
        y_train_80 = y[:train_len]
        X_test =   X[train_len:]
        y_test = y[train_len:]

    if(scale): # Scale the features in X
        scaler = MinMaxScaler()
        scaler.fit(X_train_80)
        X_train_80 = scaler.transform(X_train_80)
        X_test = scaler.transform(X_test)

    # Cross validation
    if not time_series: # k-fold cross validation
        cv = n_folds
    else: # cross validation for time series
        cv = TimeSeriesSplit(n_splits = n_folds)
    scores = cross_val_score(model, X_train_80, y_train_80, cv=cv, **kwargs)
    val_score = scores.mean() # validation score
    if regr:
        val_score = -val_score

    model.fit(X_train_80,y_train_80) # Fit the model using all the training

    # Compute training and test scores
    train_score=0
    test_score=0
    if regr:
        train_score = mean_squared_error(y_true=y_train_80, y_pred=model.predict(X_train_80))
        test_score = mean_squared_error(y_true=y_test, y_pred=model.predict(X_test))
    else:
        train_score = accuracy_score(y_true=y_train_80, y_pred=model.predict(X_train_80))
        test_score = accuracy_score(y_true=y_test, y_pred=model.predict(X_test))

    return train_score, val_score, test_score # Return a triple


def compute_bias_variance_error(X, y, model, scale=False, N_TESTS = 20, sample_size=0.67):
    """
    Compute the bias^2-variance-error scores for the given model on the given dataset.

    These measures are computed in an approximate way, using `N_TESTS` random samples of size `sample_size` from the
    dataset.

    Parameters
    ----------
    X: np.array
        Two-dimensional np.array, containing the explanatory features of the dataset.
    y: np.array
        Mono dimensional np.array, containing the response feature of the dataset.
    model: sklearn.base.BaseEstimator
        Model to evaluate.
    scale: bool
        Indicates whether to scale or not the features in `X`.
        (The scaling is performed using the sklearn MinMaxScaler).
    N_TESTS: int
        Number of samples that are made in order to compute the measures.
    sample_size: float
        Decimal number between 0 and 1, which indicates the proportion of the sample.

    Returns
    ----------
    bias: float
    variance: float
    error: float
    """

    # Scale the features in `X`
    if(scale):
        scaler = MinMaxScaler()
        scaler.fit(X)
        X = scaler.transform(X)

    # Vector 'vector_ypred': at the beginning is a list of lists (i.e. two dimensional list).
    # In the end it will be a matrix which has as many rows as `N_TESTS` (each row corresponds to a sample) and as many
    # columns as the number of instances in `X` (each column is a point of the dataset).
    # Row 'i' --> there are the predictions made by the model on the sample 'i' using all the dataset points.
    # Column 'j' --> there are the predictions made by the model on the point 'j' using all the `N_TESTS` samples.
    vector_ypred = []

    # Iterate through N_TESTS. At each iteration extract a new sample and fit the model on it.
    for i in range(N_TESTS):
        # Extract a new sample (sample 'i')
        Xs, ys = resample(X,y, n_samples=int(sample_size*len(y)) )

        # Fit the model on this sample 'i'
        model.fit(Xs,ys)

        # Add the predictions made by the model on all the dataset points
        vector_ypred.append(list(model.predict(X)))

    vector_ypred = np.array(vector_ypred) # Transform into numpy array

    # Vector that has as many elements as the dataset points, and for each of them it has the associated bias^2 computed on
    # the `N_TEST` samples.
    vector_bias = (y - np.mean(vector_ypred, axis=0))**2

    # Vector that has as many elements as the dataset points, and for each of them it has the associated variance computed on
    # the `N_TEST` samples.
    vector_variance = np.var(vector_ypred, axis=0)

    # Vector that has as many elements as the dataset points, and for each of them it has the associated error computed on
    # the `N_TEST` samples.
    vector_error = np.sum((vector_ypred - y)**2, axis=0)/N_TESTS

    bias = np.mean(vector_bias) # Total bias^2 of the model
    variance = np.mean(vector_variance) # Total variance of the model
    error = np.mean(vector_error) # Total error of the model

    return bias,variance,error # Return a triple


def plot_predictions(X, y, model, scale=False, test_size=0.2, plot_type=0, xvalues=None, xlabel="Index",
                     title="Actual vs Predicted values", figsize=(6,6)):
    """
    Plot the predictions made by the given model on the given dataset, versus its actual values.

    The dataset is split into training-test sets: the former is used to train the `model`, on the latter the predictions are
    made.

    Parameters
    ----------
    X: np.array
        Two-dimensional np.array, containing the explanatory features of the dataset.
    y: np.array
        Mono dimensional np.array, containing the response feature of the dataset.
    model: sklearn.base.BaseEstimator
        Model used to make the predictions.
    scale: bool
        Indicates whether to scale or not the features in `X`.
        (The scaling is performed using the sklearn MinMaxScaler).
    test_size: float
        Decimal number between 0 and 1, which indicates the proportion of the test set.
    plot_type: int
        Indicates the type of the plot.
            - 0 -> In the same plot two different curves are drawn: the first has on the x axis `xvalues` and on the y axis
                   the actual values (i.e. `y`); the second has on the x axis `xvalues` and on the y axis the computed
                   predicted values.
            - 1 -> On the x axis the actual values are put, on the y axis the predicted ones.
    xvalues: list (in general, iterable)
        Values that have to be put in the x axis of the plot.
        (It's used only if `plot_type` is 0).
    xlabel: str
        Label of the x axis of the plot.
        (It's used only if `plot_type` is 0).
    title: str
        Title of the plot.
    figsize: tuple
        Two dimensions of the plot.

    Returns
    ----------
    matplotlib.axes.Axes
        The matplotlib Axes where the plot has been made.

    Notes
    ----------
    The splitting of the datasets into the training-test sets is simply made by dividing the dataset into two contiguous
    sequences.
    I.e. it is the same technique used usually when the dataset is a time series dataset. (This is done in order to simplify
    the visualization).
    For this reason, typically this function is applied on time series datasets.
    """

    train_len = int(X.shape[0]*(1-test_size))
    X_train_80 = X[:train_len]
    y_train_80 = y[:train_len]
    X_test =   X[train_len:]
    y_test = y[train_len:]

    if(scale): # Scale the features in X
        scaler = MinMaxScaler()
        scaler.fit(X_train_80)
        X_train_80 = scaler.transform(X_train_80)
        X_test = scaler.transform(X_test)

    model.fit(X_train_80,y_train_80) # Fit using all the training set

    predictions = model.predict(X_test)

    fig, ax = plt.subplots(figsize=figsize)

    if plot_type==0:
        if xvalues is None:
            xvalues=range(len(X))
        ax.plot(xvalues,y, 'o:', label='actual values')
        ax.plot(xvalues[train_len:],predictions, 'o:', label='predicted values')
        ax.legend()
    elif plot_type==1:
        ax.plot(y[train_len:],predictions,'o')
        ax.plot([0, 1], [0, 1], 'r-',transform=ax.transAxes)
        xlabel="Actual values"
        ax.set_ylabel("Predicted values")

    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid()

    return ax


def _plot_TrainVal_values(xvalues, train_val_scores, plot_train, xlabel, title, figsize=(6,6), bar=False):
    """
    Plot the given list of training-validation scores.

    This function is an auxiliary function for the model selection functions. It's meant to be private in the
    module.

    Parameters
    ----------
    xvalues: list (in general iterable)
        Values to put in the x axis of the plot.
    train_val_scores: np.array
        Two dimensional np.array, containing two columns: the first contains the trainining scores, the second the validation
        scores.
        Basically, it is a list of training-validation scores.
    plot_train: bool
        Indicates whether to plot also the training scores or to plot only the validation ones.
    xlabel: str
        Label of the x axis.
    title: str
        Title of the plot.
    figsize: tuple
        Two dimensions of the plot.
    bar: bool
        Indicates whether to plot the scores using bars or using points.
        If `bar` it's True, `xvalues` must contain string (i.e. labels).
    Returns
    ----------
    matplotlib.axes.Axes
        The matplotlib Axes where the plot has been made.
    """

    fig, ax = plt.subplots(figsize=figsize)

    if not bar: # Points
        if plot_train: # Plot also the training scores
            ax.plot(xvalues,train_val_scores[:,0], 'o:', label='Train')
        ax.plot(xvalues,train_val_scores[:,1], 'o:', label='Validation') # Validation scores
    else: # Bars
        if plot_train: # Plot also the training scores
            x = np.arange(len(xvalues))  # The label locations
            width = 0.35  # The width of the bars
            ax.bar(x-width/2,train_val_scores[:,0], width=width, label='Train')
            ax.bar(x+width/2,train_val_scores[:,1], width=width, label='Validation') # Validation scores
            ax.set_xticks(x)
            ax.set_xticklabels(xvalues)
        else:
            ax.bar(xvalues,train_val_scores[:,1],label='Validation')


    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid()
    ax.legend()

    return ax




#----------------------------------------------------------------------------------------------------------------------------
# FUNCTIONS THAT PERFORM THE MODEL SELECTION WITH RESPECT TO A SINGLE DATASET


def hyperparameter_validation(X, y, model, hyperparameter, hyperparameter_values, scale=False, test_size=0.2,
                              time_series=False, random_state=123, n_folds=5, regr=True, plot=False, plot_train=False,
                              xvalues=None, xlabel=None, title="Hyperparameter validation", figsize=(6,6), **kwargs):
    """
    Select the best value for the specified hyperparameter of the specified model on the given dataset.

    In other words, perform the tuning of the `hyperparameter` among the values in `hyperparameter_values`.

    This selection is made using the validation score (i.e. the best hyperparameter value is the one with the best validation
    score).
    The validation score is computed by splitting the dataset into the training-test sets and then by applying the cross
    validation on the training set.
    Additionally, the training and test scores are also computed.

    Optionally, the validation scores of the `hyperparameter_values` can be plotted, making a graphical visualization of the
    selection.

    Parameters
    ----------
    X: np.array
        Two-dimensional np.array, containing the explanatory features of the dataset.
    y: np.array
        Mono dimensional np.array, containing the response feature of the dataset.
    model: sklearn.base.BaseEstimator
        Model which has the specified `hyperparameter`.
    hyperparameter: str
        The name of the hyperparameter that has to be validated.
    hyperparameter_values: list
        List of values for `hyperparameter` that have to be taken into account in the selection.
    scale: bool
        Indicates whether to scale or not the features in `X`.
        (The scaling is performed using the sklearn MinMaxScaler).
    test_size: float
        Decimal number between 0 and 1, which indicates the proportion of the test set.
    time_series: bool
        Indicates if the given dataset is a time series dataset (i.e. dataset indexed by days).
        (This affects the computing of the validation score).
    random_state: int
        Used in the training-test splitting of the dataset.
    n_folds: int
        Indicates how many folds are made in order to compute the k-fold cross validation.
        (It's used only if `time_series` is False).
    regr: bool
        Indicates if it's either a regression or a classification problem.
    plot: bool
        Indicates whether to plot or not the validation score values.
    plot_train: bool
        Indicates whether to plot also the training scores.
        (It's considered only if `plot` is True).
    xvalues: list (in general, iterable)
        Values that have to be put in the x axis of the plot.
    xlabel: str
        Label of the x axis of the plot.
    title: str
        Title of the plot.
    figsize: tuple
        Two dimensions of the plot.

    Returns
    ----------
    train_val_scores: np.array
        Two dimensional np.array, containing two columns: the first contains the training scores, the second the validation
        scores.
        It has as many rows as the number of values in `hyperparameter_values` (i.e. number of values to be tested).
    best_index: int
        Index of `hyperparameter_values` that indicates which is the best hyperparameter value.
    test_score: float
        Test score associated with the best hyperparameter value.
    ax: matplotlib.axes.Axes
        The matplotlib Axes where the plot has been made.
        If `plot` is False, then it is None.

    Notes
    ----------
    - If `regr` is True, the validation scores are errors (MSE, i.e. Mean Squared Errors): this means that the best
      hyperparameter value is the one associated with the minimum validation score.
      Otherwise, the validation scores are accuracies: this means that the best hyperparameter value is the one associated
      with the maximum validation score.
    - If `time_series` is False, the training-test splitting of the dataset is made randomly. In addition, the cross
      validation strategy performed is the classic k-fold cross validation: the number of folds is specified by `n_folds`.
      Otherwise, if `time_series` is True, the training-test sets are simply obtained by splitting the dataset into two
      contiguous parts. In addition, the cross validation strategy performed is the sklearn TimeSeriesSplit.
    """

    param_grid = {hyperparameter:hyperparameter_values} # Create the hyperparameter grid
    # Call the function for the validation of an arbitrary number of hyperparameters
    params, grid_search, test_score = hyperparameters_validation(X, y, model, param_grid, scale=scale,
                                                                                  test_size=test_size,
                                                                                  time_series=time_series,
                                                                                  random_state=random_state, n_folds=n_folds,
                                                                                  regr=regr, **kwargs)

    ax = None

    # if(plot): # Make the plot
    #     if not xvalues: # Default values on the x axis
    #         xvalues = hyperparameter_values
    #     if not xlabel: # Default label on the x axis
    #         xlabel = hyperparameter
    #     ax = _plot_TrainVal_values(xvalues, train_val_scores, plot_train, xlabel, title, figsize)

    return params, grid_search, test_score


def hyperparameters_validation(X, y, model, param_grid, scale=False, test_size=0.2, time_series=False, random_state=123,
                               n_folds=5, regr=True, **kwargs):
    """
    Select the best combination of values for the specified hyperparameters of the specified model on the given dataset.

    In other words, perform the tuning of multiple hyperparameters.
    The parameter `param_grid` is a dictionary that indicates which are the specified hyperparameters and what are the
    associated values to test.

    All the possible combinations of values are tested, in an exhaustive way (i.e. grid search).

    This selection is made using the validation score (i.e. the best combination of hyperparameters values is the one with
    the best validation score).
    The validation score is computed by splitting the dataset into the training-test sets and then by applying the cross
    validation on the training set.
    Additionally, the training and test scores are also computed.

    Parameters
    ----------
    X: np.array
        Two-dimensional np.array, containing the explanatory features of the dataset.
    y: np.array
        Mono dimensional np.array, containing the response feature of the dataset.
    model: sklearn.base.BaseEstimator
        Model which has the specified hyperparameters.
    param_grid: dict
        Dictionary which has as keys the names of the specified hyperparameters and as values the associated list of
        values to test.
    scale: bool
        Indicates whether to scale or not the features in `X`.
        (The scaling is performed using the sklearn MinMaxScaler).
    test_size: float
        Decimal number between 0 and 1, which indicates the proportion of the test set.
    time_series: bool
        Indicates if the given dataset is a time series dataset (i.e. dataframe indexed by days).
        (This affects the computing of the validation score).
    random_state: int
        Used in the training-test splitting of the dataset.
    n_folds: int
        Indicates how many folds are made in order to compute the k-fold cross validation.
        (It's used only if `time_series` is False).
    regr: bool
        Indicates if it's either a regression or a classification problem.

    Returns
    ----------
    params: list
        List which enumerates all the possible combinations of hyperparameters values.
        It's a list of dictionaries: each dictionary represents a specific combination of hyperparameters values. (It's a
        dictionary which has as keys the hyperparameters names and as values the specific associated values of that combination).
    train_val_scores: np.array
        Two dimensional np.array, containing two columns: the first contains the training scores, the second the validation
        scores.
        It has as many rows as the number of possible combinations of the hyperparameters values.
        (It has as many rows as the elements of `params`).
    best_index: int
        Index of `params` that indicates which is the best combination of hyperparameters values.
    test_score: float
        Test score associated with the best combination of hyperparameters values.

    Notes
    ----------
    - If `regr` is True, the validation scores are errors (MSE, i.e. Mean Squared Errors): this means that the best
      combination of hyperparameters values is the one associated with the minimum validation score.
      Otherwise, the validation scores are accuracies: this means that the best combination of hyperparameters values is the
      one associated with the maximum validation score.
    - If `time_series` is False, the training-test splitting of the dataset is made randomly. In addition, the cross
      validation strategy performed is the classic k-fold cross validation: the number of folds is specified by `n_folds`.
      Otherwise, if `time_series` is True, the training-test sets are simply obtained by splitting the dataset into two
      contiguous parts. In addition, the cross validation strategy performed is the sklearn TimeSeriesSplit.
    """

    # if regr:
    #     scoring="neg_mean_squared_error"
    # else:
    #     scoring="accuracy"

    # Split into training-test sets
    if not time_series : # Random splitting
        X_train_80, X_test, y_train_80, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    else: # Time series splitting
        train_len = int(X.shape[0]*(1-test_size))
        X_train_80 = X[:train_len]
        y_train_80 = y[:train_len]
        X_test =   X[train_len:]
        y_test = y[train_len:]

    if(scale): # Scale the features in `X`
        scaler = MinMaxScaler()
        scaler.fit(X_train_80)
        X_train_80 = scaler.transform(X_train_80)
        X_test = scaler.transform(X_test)

    # Cross validation strategy
    if not time_series: # The strategy is the classic k-fold cross validation
        cv = n_folds
    else: # Time series cross validation strategy
        cv = TimeSeriesSplit(n_splits = n_folds)

    # Grid search
    grid_search = GridSearchCV(model,param_grid,cv=cv,return_train_score=True, **kwargs)
    grid_search.fit(X_train_80,y_train_80)

    logging.info(grid_search.cv_results_.keys())
    params = grid_search.cv_results_["params"] # List of all the possible combinations of hyperparameters values
    # List where for all the possible combinations of hyperparameters values there is the associated training score
    train_scores = grid_search.cv_results_["mean_train_score"]
    # List where for all the possible combinations of hyperparameters values there is the associated validation score
    val_scores = grid_search.cv_results_["mean_test_score"]
    # Index of `params`, corresponding to the best combination of hyperparameters values
    best_index = grid_search.best_index_
    # Model with the best combination of hyperparameters values
    best_model = grid_search.best_estimator_

    if regr: # The scores are negative: moltiply by -1
        train_scores = train_scores*(-1)
        val_scores = val_scores*(-1)
    train_val_scores = np.concatenate((train_scores.reshape(-1,1), val_scores.reshape(-1,1)), axis=1)

    # # Fit the best model on all the training set
    # best_model.fit(X_train_80,y_train_80)

    # Compute the test score of the best model
    test_score=grid_search.score(X_test,y_test)
    # if regr:
    #     test_score = mean_squared_error(y_true=y_test, y_pred=best_model.predict(X_test))
    # else:
    #     if kwargs.get("scoring")=="accuracy":
    #         # test_score = best_model.score(X_test,y_test)
    #         test_score = accuracy_score(y_true=y_test, y_pred=best_model.predict(X_test))
    #     elif kwargs.get("scoring")=="f1":
    #         test_score = f1_score(y_true=y_test, y_pred=best_model.predict(X_test))
    #     elif kwargs.get("scoring")=="precision":
    #         test_score = precision_score(y_true=y_test, y_pred=best_model.predict(X_test))
    #     elif kwargs.get("scoring")=="recall":
    #         test_score = recall_score(y_true=y_test, y_pred=best_model.predict(X_test))
    #     elif kwargs.get("scoring")=="roc_auc":
    #         test_score = roc_auc_score(y_true=y_test, y_score=best_model.predict_proba(X_test))
    #     else:
    #         # use accuracy as default
    #         test_score = accuracy_score(y_true=y_test, y_pred=best_model.predict(X_test))

    return params, grid_search, test_score

def plot_learning_curve(
    estimator,
    X,
    y,
    title,
    axes=None,
    ylim=None,
    scoring=None,
    cv=5,
    n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 5),
):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator : estimator instance
        An estimator instance implementing `fit` and `predict` methods which
        will be cloned for each validation.

    title : str
        Title for the chart.

    X : array-like of shape (n_samples, n_features)
        Training vector, where ``n_samples`` is the number of samples and
        ``n_features`` is the number of features.

    y : array-like of shape (n_samples) or (n_samples, n_features)
        Target relative to ``X`` for classification or regression;
        None for unsupervised learning.

    axes : array-like of shape (3,), default=None
        Axes to use for plotting the curves.

    ylim : tuple of shape (2,), default=None
        Defines minimum and maximum y-values plotted, e.g. (ymin, ymax).

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like of shape (n_ticks,)
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the ``dtype`` is float, it is regarded
        as a fraction of the maximum size of the training set (that is
        determined by the selected validation method), i.e. it has to be within
        (0, 1]. Otherwise it is interpreted as absolute sizes of the training
        sets. Note that for classification the number of samples usually have
        to be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
        estimator,
        X,
        y,
        scoring=estimator.scoring if scoring is None else scoring,
        cv=cv,
        n_jobs=n_jobs,
        train_sizes=train_sizes,
        return_times=True,
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    axes[0].fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )
    axes[0].plot(
        train_sizes, train_scores_mean, "o-", color="r", label="Training score"
    )
    axes[0].plot(
        train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score"
    )
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, "o-")
    axes[1].fill_between(
        train_sizes,
        fit_times_mean - fit_times_std,
        fit_times_mean + fit_times_std,
        alpha=0.1,
    )
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    fit_time_argsort = fit_times_mean.argsort()
    fit_time_sorted = fit_times_mean[fit_time_argsort]
    test_scores_mean_sorted = test_scores_mean[fit_time_argsort]
    test_scores_std_sorted = test_scores_std[fit_time_argsort]
    axes[2].grid()
    axes[2].plot(fit_time_sorted, test_scores_mean_sorted, "o-")
    axes[2].fill_between(
        fit_time_sorted,
        test_scores_mean_sorted - test_scores_std_sorted,
        test_scores_mean_sorted + test_scores_std_sorted,
        alpha=0.1,
    )
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    plt.show()

def plot_confusion_matrix(y_true, y_pred, classes, title, regr=True, cmap=plt.cm.Blues, normalize='true'):
    """
    Plot the confusion matrix of the model.

    Parameters
    ----------
    y_true: np.array
        The true target of the dataset.
    y_pred: np.array
        The predicted target of the dataset.
    classes: np.array
        The classes of the dataset.
    title: str
        The title of the plot.
    regr: bool, optional
        If True, the confusion matrix is computed as the mean of the errors (MSE, i.e. Mean Squared Errors).
        Otherwise, the confusion matrix is computed as the mean of the accuracies (i.e. Accuracy).
        By default, it is True.
    cmap: matplotlib.colors.Colormap, optional
        The colormap used to plot the confusion matrix.

    Returns
    -------
    None
    """

    # Compute confusion matrix
    disp = ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=classes,
        # cmap=cmap,
        normalize=normalize,
    )
    disp.ax_.set_title(title)
    # plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def plot_precision_recall_curve(y_true, y_score, title, cmap=plt.cm.Blues):
    """
    Plot the precision recall curve of the model.

    Parameters
    ----------
    y_true: np.array
        The true target of the dataset.
    y_score: np.array
        The predicted target of the dataset.
    title: str
        The title of the plot.
    cmap: matplotlib.colors.Colormap, optional
        The colormap used to plot the confusion matrix.

    Returns
    -------
    None
    """

    # Compute confusion matrix
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    average_precision = average_precision_score(y_true, y_score)

    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(title)
    plt.show()

def plot_roc_curve(y_true, y_score, title, cmap=plt.cm.Blues):
    """
    Plot the ROC curve of the model.

    Parameters
    ----------
    y_true: np.array
        The true target of the dataset.
    y_score: np.array
        The predicted target of the dataset.
    title: str
        The title of the plot.
    cmap: matplotlib.colors.Colormap, optional
        The colormap used to plot the confusion matrix.

    Returns
    -------
    None
    """

    # Compute ROC curve and ROC area
    fpr, tpr, thresholds = roc_curve(y_true=y_true, y_score=y_score)
    roc_auc = auc(x=fpr, y=tpr)

    # Plot ROC curve
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()

def model_evaluation(grid_search, X, y, cv=5, n_jobs=4, train_sizes=np.linspace(.1, 1.0, 5), title=None, prediction_threshold=0.5):
    """
    Get extensive and exhaustive, report including the learning curve, the confusion matrix and the ROC curve.

    Parameters
    ----------
    grid_search: sklearn.model_selection.GridSearchCV
        The grid search object.
    X: np.array
        The input of the dataset.
    y: np.array
        The target of the dataset.
    cv: int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
            - None, to use the default 3-fold cross-validation,
            - integer, to specify the number of folds.
            - :term:`CV splitter`,
            - An iterable yielding (train, test) splits as arrays of indices.
            For integer/None inputs, if ``y`` is binary or multiclass,
            :class:`StratifiedKFold` used. If the estimator is a classifier
            or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.
            Refer :ref:`User Guide <cross_validation>` for the various
            cross-validation strategies that can be used here.
    n_jobs: int, optional
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>` for more details.
    train_sizes: np.array, optional
        The training set sizes to plot.
        If None, default to np.linspace(0.1, 1.0, 5)

    Returns
    -------
    None
    """

    # Get extensive and exhaustive report
    # report = classification_report(y_true=y, y_pred=grid_search.predict(X=X), output_dict=True)
    # print("\nExtensive and exhaustive report:")
    # for key in report:
    #     # print("{:<20}: {:.2f}".format(key, report[key]))
    #     print(f"{key}: {report[key]}")
    y_pred = grid_search.predict_proba(X=X)
    y_pred = np.array([1 if y_pred[i][1] >= prediction_threshold else 0 for i in range(len(y_pred))])
    print(classification_report(y_true=y, y_pred=y_pred))

    # Get learning curve
    # plot_learning_curve(grid_search=grid_search, X=X, y=y, title=title, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, regr=False)

    # Get confusion matrix
    plot_confusion_matrix(y_true=y, y_pred=y_pred, classes=np.unique(y), title=f"Confusion matrix of {title}", regr=False)

    # Get ROC curve
    # check if the estimator has a best_estimator_ method
    if hasattr(grid_search, 'best_estimator_'):
        # check if the estimator has a predict_proba method
        if hasattr(grid_search.best_estimator_, 'predict_proba'):
            plot_roc_curve(y_true=y, y_score=grid_search.best_estimator_.predict_proba(X=X)[:, 1], title=f"ROC curve of {title}")
            plot_precision_recall_curve(y_true=y, y_score=grid_search.best_estimator_.predict_proba(X=X)[:, 1], title=f"Precision-recall curve of {title}")
        elif hasattr(grid_search.best_estimator_, 'decision_function'):
            plot_roc_curve(y_true=y, y_score=grid_search.decision_function(X=X), title=f"ROC curve of {title}")
            plot_precision_recall_curve(y_true=y, y_score=grid_search.decision_function(X=X), title=f"Precision-recall curve of {title}")
    else:
        # check if the estimator has a predict_proba method
        if hasattr(grid_search, 'predict_proba'):
            plot_roc_curve(y_true=y, y_score=grid_search.predict_proba(X=X)[:, 1], title=f"ROC curve of {title}")
            plot_precision_recall_curve(y_true=y, y_score=grid_search.predict_proba(X=X)[:, 1], title=f"Precision-recall curve of {title}")
        elif hasattr(grid_search, 'decision_function'):
            plot_roc_curve(y_true=y, y_score=grid_search.decision_function(X=X), title=f"ROC curve of {title}")
            plot_precision_recall_curve(y_true=y, y_score=grid_search.decision_function(X=X), title=f"Precision-recall curve of {title}")
        

def models_evaluation(grid_searches, X, y, cv=5, n_jobs=4, train_sizes=np.linspace(.1, 1.0, 5), titles=None):
    """
    Get extensive and exhaustive, report including the learning curve, the confusion matrix and the ROC curve for all the models.

    Parameters
    ----------
    grid_searches: list
        The grid search objects.
    X: np.array
        The input of the dataset.
    y: np.array
        The target of the dataset.
    cv: int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
            - None, to use the default 3-fold cross-validation,
            - integer, to specify the number of folds.
            - :term:`CV splitter`,
            - An iterable yielding (train, test) splits as arrays of indices.
            For integer/None inputs, if ``y`` is binary or multiclass,
            :class:`StratifiedKFold` used. If the estimator is a classifier
            or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.
            Refer :ref:`User Guide <cross_validation>` for the various
            cross-validation strategies that can be used here.
    n_jobs: int, optional
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>` for more details.
    train_sizes: np.array, optional
        The training set sizes to plot.
        If None, default to np.linspace(0.1, 1.0, 5)

    Returns
    -------
    None
    """

    # Get extensive and exhaustive report for all the models
    for i, grid_search in enumerate(grid_searches):
        model_evaluation(grid_search=grid_search, X=X, y=y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, title=titles[i])

def models_building(X, y, model_paramGrid_list, scale_list=None, test_size=0.2, time_series=False, random_state=123,
                      n_folds=5, regr=True, show_progress=False, logger=None, plot=False, plot_train=False, xvalues=None, xlabel="Models",
                      title="Models validation", figsize=(6,6), **kwargs):
    """
    Select the best model on the given dataset.

    The parameter `model_paramGrid_list` is the list of the models to test. It also contains, for each model, the grid of
    hyperparameters that have to be tested on that model (i.e. the grid which contains the values to test for each
    specified hyperparameter of the model).
    (That grid has the same structure as the `param_grid` parameter of the function `hyperparameters_validation`. See
    `hyperparameters_validation`).

    For each specified model, the best combination of hyperparameters values is selected in an exhaustive  way (i.e. grid
    search).
    Actually, the function `hyperparameters_validation` is used.
    (See `hyperparameters_validation`).

    The selection of the best model is made using the validation score (i.e. the best model is the one with the best
    validation score).
    The validation score is computed by splitting the dataset into the training-test sets and then by applying the cross
    validation on the training set.
    Additionally, the training and test scores are also computed.

    Optionally, the validation scores of the different models can be plotted, making a graphical visualization of the
    selection.

    Parameters
    ----------
    X: np.array
        Two-dimensional np.array, containing the explanatory features of the dataset.
    y: np.array
        Mono dimensional np.array, containing the response feature of the dataset.
    model_paramGrid_list: list
        List that specifies the models and the relative grids of hyperparameters to be tested.
        It's a list of triples (i.e. tuples), where each triple represents a model:
            - the first element is a string, which is a mnemonic name of that model;
            - the second element is the sklearn model;
            - the third element is the grid of hyperparameters to test for that model. It's a dictionary, with the same
              structure of the parameter `param_grid` of the function `hyperparameters_validation`.
    scale_list: list or bool
        List of booleans, which has as many elements as the models to test (i.e. as the elements of the
        `model_paramGrid_list` list).
        This list indicates, for each different model, if the features in `X` have to be scaled or not.
        `scale_list` can be None or False: in this case the `X` features aren't scaled for any model. `scale_list` can be
        True: in this case the `X` features are scaled for all the models.
    test_size: float
        Decimal number between 0 and 1, which indicates the proportion of the test set.
    time_series: bool
        Indicates if the given dataset is a time series dataset (i.e. dataset indexed by days).
        (This affects the computing of the validation score).
    random_state: int
        Used in the training-test splitting of the dataset.
    n_folds: int
        Indicates how many folds are made in order to compute the k-fold cross validation.
        (It's used only if `time_series` is False).
    regr: bool
        Indicates if it's either a regression or a classification problem.
    plot: bool
        Indicates whether to plot or not the validation score values.
    plot_train: bool
        Indicates whether to plot also the training scores.
        (It's considered only if `plot` is True).
    xvalues: list (in general, iterable)
        Values that have to be put in the x axis of the plot.
    xlabel: str
        Label of the x axis of the plot.
    title: str
        Title of the plot.
    figsize: tuple
        Two dimensions of the plot.

    Returns
    ----------
    models_train_val_score: np.array
        Two dimensional np.array, containing two columns: the first contains the training scores, the second the validation
        scores.
        It has as many rows as the number of models to test (i.e. number of elements in the `model_paramGrid_list` list).
    models_best_params: list
        List which indicates, for each model, the best combination of the hyperparameters values for that model.
        It has as many elements as the models to test (i.e. as the elements of the `model_paramGrid_list` list), and it
        contains dictionaries: each dictionary represents the best combination of the hyperparameters values for the
        associated model.
    best_index: int
        Index of `model_paramGrid_list` that indicates which is the best model.
    test_score: float
        Test score associated with the best model.
    ax: matplotlib.axes.Axes
        The matplotlib Axes where the plot has been made.
        If `plot` is False, then it is None.

    See also
    ----------
    hyperparameters_validation:
        select the best combination of values for the specified hyperparameters of the specified model on the given dataset.

    Notes
    ----------
    - If `regr` is True, the validation scores are errors (MSE, i.e. Mean Squared Errors): this means that the best
      model is the one associated with the minimum validation score.
      Otherwise, the validation scores are accuracies: this means that the best model is the one associated with the
      maximum validation score.
    - If `time_series` is False, the training-test splitting of the dataset is made randomly. In addition, the cross
      validation strategy performed is the classic k-fold cross validation: the number of folds is specified by `n_folds`.
      Otherwise, if `time_series` is True, the training-test sets are simply obtained by splitting the dataset into two
      contiguous parts. In addition, the cross validation strategy performed is the sklearn TimeSeriesSplit.
    """

    if not scale_list: # `scale_list` is either None or False
        scale_list = [False]*len(model_paramGrid_list)
    elif scale_list is True: # `scale_list` is True
        scale_list = [True]*len(model_paramGrid_list)

    # # Numpy matrix (np.array) which has as many rows as the models and which has two columns, one for the training scores and
    # # the other for the validation scores. At the beginning it is a list of tuples.
    # models_train_val_score = []
    # # List which has as many elements as the models: for each model there is the dictionary of the best combination of
    # # hyperparameters values.
    # models_best_params = []
    # # List which has as many elements as the models: for each model there is the test score (associated with the best
    # # combination of hyperparameters values).
    # models_test_score = []
    models = []

    for i,triple in enumerate(model_paramGrid_list): # Iterate through all the cuples model-param_grid
        model_name, model,param_grid = triple

        # Apply the grid search on model-param_grid
        params, grid_search, test_score = hyperparameters_validation(X, y, model, param_grid,
                                                                                      scale=scale_list[i],
                                                                                      test_size=test_size,
                                                                                      time_series=time_series,
                                                                                      random_state=random_state,
                                                                                      n_folds=n_folds, regr=regr, **kwargs)
        models.append((model_name, model, params, grid_search, test_score))
        # models_train_val_score.append(tuple(train_val_scores[best_index])) # Add the row for that model
        # models_best_params.append(params[best_index]) # Add the element for that model
        # models_test_score.append(test_score) # Add the element for that model
        if show_progress:
            logger.log(logging.INFO, f'Model {i+1}/{len(model_paramGrid_list)} done : {model_paramGrid_list[i][0]}')
            # logger.log(logging.INFO, f'\tBest params: {models_best_params[-1]}')
            # logger.log(logging.INFO, f'\tTrain score: {models_train_val_score[-1][0]}')
            # logger.log(logging.INFO, f'\tVal score: {models_train_val_score[-1][1]}')
            # logger.log(logging.INFO, f'\tTest score: {models_test_score[-1]}')

    # models_train_val_score = np.array(models_train_val_score) # Transform into numpy matrix (i.e. np.array)

    # # Find the best index (i.e. the best model)
    # if regr:
    #     best_index = np.argmin(models_train_val_score,axis=0)[1]
    # else:
    #     best_index = np.argmax(models_train_val_score,axis=0)[1]

    # # Test score of the best model
    # test_score = models_test_score[best_index]
    # logger.log(logging.INFO, f'Best model: {best_index+1}/{len(model_paramGrid_list)} : {model_paramGrid_list[best_index][0]}')

    # ax = None
    # if(plot): # Make the plot
    #     if not xvalues: # Default values for the x axis
    #         xvalues = [model_paramGrid_list[i][0] for i in range(len(model_paramGrid_list))]
    #     ax = _plot_TrainVal_values(xvalues, models_train_val_score, plot_train, xlabel, title, figsize, bar=True)
    
    logger.log(logging.INFO, '-----------------------------------------------------------')
    logger.log(logging.INFO, models)
    models.sort(key=lambda x: x[-1], reverse=True)
    return models




#----------------------------------------------------------------------------------------------------------------------------
# FUNCTIONS THAT PERFORM THE MODEL SELECTION WITH RESPECT TO MULTIPLE DATASETS


def datasets_hyperparameter_validation(dataset_list, model, hyperparameter, hyperparameter_values, scale=False,
                                       test_size=0.2, time_series=False, random_state=123, n_folds=5, regr=True, logger=None, plot=False,
                                       plot_train=False, xvalues=None, xlabel="Datasets", title="Datasets validation",
                                       figsize=(6,6) ,verbose=False, figsize_verbose=(6,6), **kwargs):
    """
    Select the best dataset and the best value for the specified hyperparameter of the specified model (i.e. select the best
    couple dataset-hyperparameter value).

    For each dataset in `dataset_list`, all the specified values `hyperparameter_values` are tested for the specified
    `hyperparameter` of `model`.
    In other words, on each dataset the tuning of `hyperparameter` is performed: in fact, on each dataset, the function
    `hyperparameter_validation` is applied. (See `hyperparameter_validation`).
    In the end, the best couple dataset-hyperparameter value is selected.

    Despite the fact that a couple dataset-hyperparameter value is selected, the main viewpoint is focused with respect to
    the datasets. It's a validation focused on the datasets.
    In fact, first of all, for each dataset the hyperparameter tuning is performed: in this way the best value is selected
    and its relative score is associated with the dataset (i.e. it's the score of the dataset). (In other words, on each
    dataset the function `hyperparameter_validation` is applied). Finally, after that, the best dataset is selected.
    It's a two-levels selection.

    This selection is made using the validation score (i.e. the best couple dataset-hyperparameter value is the one with the
    best validation score).
    The validation score is computed by splitting each dataset into the training-test sets and then by applying the cross
    validation on the training set.
    Additionally, the training and test scores are also computed.

    Optionally, the validation scores of the datasets can be plotted, making a graphical visualization of the dataset
    selection. This is the 'main' plot.
    Moreover, still optionally, the 'secondary' plots can be done: for each dataset, the validation scores of the
    `hyperparameter_values` are plotted, making a graphical visualization of the hyperparameter tuning on that dataset.
    (As the plot made by the `hyperparameter_validation` function).

    Parameters
    ----------
    dataset_list: list
        List of couples, where each couple is a dataset.
            - The first element is X, the two-dimensional np.array containing the explanatory features of the dataset.
            - The second element is y, the mono dimensional np.array containing the response feature of the dataset.
    model: sklearn.base.BaseEstimator
        Model which has the specified `hyperparameter`.
    hyperparameter: str
        The name of the hyperparameter that has to be validated.
    hyperparameter_values: list
        List of values for `hyperparameter` that have to be taken into account in the selection.
    scale: bool
        Indicates whether to scale or not the features in 'X' (for all the datasets).
        (The scaling is performed using the sklearn MinMaxScaler).
    test_size: float
        Decimal number between 0 and 1, which indicates the proportion of the test set (for each dataset).
    time_series: bool
        Indicates if the given datasets are time series dataset (i.e. datasets indexed by days).
        (This affects the computing of the validation scores).
    random_state: int
        Used in the training-test splitting of the datasets.
    n_folds: int
        Indicates how many folds are made in order to compute the k-fold cross validation.
        (It's used only if `time_series` is False).
    regr: bool
        Indicates if it's either a regression or a classification problem.
    plot: bool
        Indicates whether to plot or not the validation score values of the datasets (i.e. this is the 'main' plot).
    plot_train: bool
        Indicates whether to plot also the training scores (both in the 'main' and 'secondary' plots).
    xvalues: list (in general, iterable)
        Values that have to be put in the x axis of the 'main' plot.
    xlabel: str
        Label of the x axis of the 'main' plot.
    title: str
        Title of the 'main' plot.
    figsize: tuple
        Two dimensions of the 'main' plot.
    verbose: bool
        If True, for each dataset are plotted the validation scores of the hyperparameter tuning (these are the 'secondary'
        plots).
        (See 'hyperparameter_validation').
    figsize_verbose: tuple
        Two dimensions of the 'secondary' plots.

    Returns
    ----------
    datasets_train_val_score: np.array
        Two dimensional np.array, containing two columns: the first contains the training scores, the second the validation
        scores.
        It has as many rows as the number of datasets to test, i.e. as the number of elements in `dataset_list`.
    datasets_best_hyperparameter_value: list
        List which has as many elements as the number of the datasets (i.e. as the number of elements in `dataset_list`). For
        each dataset, it contains the best `hyperparameter` value on that dataset.
    best_index: int
        Index of `dataset_list` that indicates which is the best dataset.
    test_score: float
        Test score associated with the best couple dataset-hyperparameter value.
    axes: list
        List of the matplotlib Axes where the plots have been made.
        Firstly, the 'secondary' plots are put (if any). And, as last, the 'main' plot is put (if any).
        If no plot has been made, `axes` is an empty list.

    See also
    ----------
    hyperparameter_validation:
        select the best value for the specified hyperparameter of the specified model on the given dataset.

    Notes
    ----------
    - If `regr` is True, the validation scores are errors (MSE, i.e. Mean Squared Errors): this means that the best
      couple dataset-hyperparameter value is the one associated with the minimum validation score.
      Otherwise, the validation scores are accuracies: this means that the best couple is the one associated with the
      maximum validation score.
    - If `time_series` is False, the training-test splitting of each dataset is made randomly. In addition, the cross
      validation strategy performed is the classic k-fold cross validation: the number of folds is specified by `n_folds`.
      Otherwise, if `time_series` is True, the training-test sets are simply obtained by splitting each dataset into two
      contiguous parts. In addition, the cross validation strategy performed is the sklearn TimeSeriesSplit.
    """

    # numpy matrix (i.e. np.array) which has as many rows as the datasets, and it has the training and validation scores as
    # columns. At the beginning it is a list.
    datasets_train_val_score = []
    # List which contains, for each dataset, the best hyperparameter value
    datasets_best_hyperparameter_value = []
    # List which contains, for each dataset, its test score (associated with the best hyperparameter value)
    datasets_test_score = []
    # List of axes
    axes = []

    for i,dataset in enumerate(dataset_list): # Iterate through all the datasets

        X,y = dataset

        # Perform the hyperparameter tuning on the current dataset
        train_val_scores, best_index, test_score, ax = hyperparameter_validation(X, y, model, hyperparameter,
                                hyperparameter_values, scale=scale, test_size=test_size, time_series=time_series,
                                random_state=random_state, n_folds=n_folds, regr=regr, plot=verbose, plot_train=plot_train,
                                xvalues=hyperparameter_values, xlabel=hyperparameter,
                                title="Dataset "+str(i)+" : hyperparameter validation", figsize=figsize_verbose, **kwargs)

        datasets_train_val_score.append(tuple(train_val_scores[best_index,:])) # Add the row related to that dataset
        datasets_best_hyperparameter_value.append(hyperparameter_values[best_index]) # Add the element related to that dataset
        datasets_test_score.append(test_score) # Add the row related to that dataset
        logger.log(logging.INFO, f"Dataset {i} : best hyperparameter value = {hyperparameter_values[best_index]}")
        if ax:
            axes.append(ax)

    datasets_train_val_score = np.array(datasets_train_val_score) # Transform into numpy

    # Find the best index, i.e. the best dataset (more precisely, the best couple dataset-hyperparameter value)
    if regr:
        best_index = np.argmin(datasets_train_val_score,axis=0)[1]
    else:
        best_index = np.argmax(datasets_train_val_score,axis=0)[1]

    # Test score of the best couple dataset-hyperparameter value
    test_score = datasets_test_score[best_index]
    logger.log(logging.INFO, f"Best dataset : {best_index} : best hyperparameter value = {datasets_best_hyperparameter_value[best_index]}, with test score = {test_score}")

    if(plot): # Make the plot
        if not xvalues: # Default values on the x axis
            xvalues = range(len(dataset_list))
        ax = _plot_TrainVal_values(xvalues,datasets_train_val_score,plot_train,xlabel,title,figsize, bar=True)
        axes.append(ax)

    return datasets_train_val_score, datasets_best_hyperparameter_value, best_index, test_score, axes


def datasets_hyperparameters_validation(dataset_list, model, param_grid, scale=False, test_size=0.2, time_series=False,
                                        random_state=123, n_folds=5, regr=True, logger=None, plot=False, plot_train=False, xvalues=None,
                                        xlabel="Datasets", title="Datasets validation",figsize=(6,6), **kwargs):
    """
    Select the best dataset and the best combination of values for the specified hyperparameters of the specified model (i.e.
    select the best couple dataset-combination of hyperparameters values).

    For each dataset in `dataset_list`, all the possible combinations of the hyperparameters values for `model` (specified
    with `param_grid`) are tested.
    In other words, on each dataset the tuning of the specified hyperparameters is performed in an exhaustive way: in fact,
    on each dataset, the function `hyperparameters_validation` is applied. (See `hyperparameters_validation`).
    In the end, the best couple dataset-combination of hyperparameters values is selected.

    Despite the fact that a couple dataset-combination of hyperparameters values is selected, the main viewpoint is focused
    with respect to the datasets. It's a validation focused on the datasets.
    In fact, first of all, for each dataset the hyperparameters tuning is performed: in this way the best combination of
    values is selected and its relative score is associated with the dataset (i.e. it's the score of the dataset). (In other
    words, on each dataset the function `hyperparameters_validation` is applied). Finally, after that, the best dataset is
    selected. It's a two-levels selection.

    This selection is made using the validation score (i.e. the best couple dataset-combination of hyperparameters values, is
    the one with best validation score).
    The validation score is computed by splitting each dataset into the training-test sets and then by applying the cross
    validation on the training set.
    Additionally, the training and test scores are also computed.

    Optionally, the validation scores of the datasets can be plotted, making a graphical visualization of the dataset
    selection.

    Parameters
    ----------
    dataset_list: list
        List of couple, where each couple is a dataset.
            - The first element is X, the two-dimensional np.array containing the explanatory features of the dataset.
            - The second element is y, the mono dimensional np.array containing the response feature of the dataset.
    model: sklearn.base.BaseEstimator
        Model which has the specified hyperparameters.
    param_grid: dict
        Dictionary which has as keys the names of the specified hyperparameters and as values the associated list of
        values to test.
    scale: bool
        Indicates whether to scale or not the features in 'X' (for all the datasets).
        (The scaling is performed using the sklearn MinMaxScaler).
    test_size: float
        Decimal number between 0 and 1, which indicates the proportion of the test set (for each dataset).
    time_series: bool
        Indicates if the given datasets are time series datasets (i.e. datasets indexed by days).
        (This affects the computing of the validation score).
    random_state: int
        Used in the training-test splitting of the datasets.
    n_folds: int
        Indicates how many folds are made in order to compute the k-fold cross validation.
        (It's used only if `time_series` is False).
    regr: bool
        Indicates if it's either a regression or a classification problem.
    plot: bool
        Indicates whether to plot or not the validation score values of the datasets.
    plot_train: bool
        Indicates whether to plot also the training scores.
        (It's considered only if `plot` is True).
    xvalues: list (in general, iterable)
        Values that have to be put in the x axis of the plot.
    xlabel: str
        Label of the x axis of the plot.
    title: str
        Title of the plot.
    figsize: tuple
        Two dimensions of the plot.

    Returns
    ----------
    datasets_train_val_score: np.array
        Two dimensional np.array, containing two columns: the first contains the training scores, the second the validation
        scores.
        It has as many rows as the number of datasets to test, i.e. as the number of elements in `dataset_list`.
    datasets_best_params: list
        List which has as many elements as the number of the datasets (i.e. as the number of elements in `dataset_list`). For
        each dataset, it contains the best combination of hyperparameters values on that dataset.
        Each combination is represented as a dictionary, with keys the hyperparameters names and values the associated
        values.
    best_index: int
        Index of `dataset_list` that indicates which is the best dataset.
    test_score: float
        Test score associated with the best couple dataset-combination of hyperparameters values.
    ax: matplotlib.axes.Axes
        The matplotlib Axes where the plot has been made.

    See also
    ----------
    hyperparameters_validation:
        select the best combination of values for the specified hyperparameters of the specified model on the given dataset.

    Notes
    ----------
    - If `regr` is True, the validation scores are errors (MSE, i.e. Mean Squared Errors): this means that the best
      couple dataset-combination of hyperparameters values is the one associated with the minimum validation score.
      Otherwise, the validation scores are accuracies: this means that the best couple is the one associated with the
      maximum validation score.
    - If `time_series` is False, the training-test splitting of each dataset is made randomly. In addition, the cross
      validation strategy performed is the classic k-fold cross validation: the number of folds is specified by `n_folds`.
      Otherwise, if `time_series` is True, the training-test sets are simply obtained by splitting each dataset into two
      contiguous parts. In addition, the cross validation strategy performed is the sklearn TimeSeriesSplit.
    """

    # numpy matrix (i.e. np.array) which has as many rows as the datasets, and it has the training and validation scores as
    # columns . At the beginning it is a list.
    datasets_train_val_score = []
    # List which contains, for each dataset, the best combination of hyperparameters values (i.e. a dictionary)
    datasets_best_params = []
    # List which contains, for each dataset, its test score (associated to the best combination of hyperparameters values)
    datasets_test_score = []

    for i, (X,y) in enumerate(dataset_list): # Iterate through all the datasets

        # Perform the exaustive hyperparameters tuning on the current dataset
        params, train_val_scores, best_index, test_score = hyperparameters_validation(X, y, model, param_grid, scale=scale,
                                                                                      test_size=test_size,
                                                                                      time_series=time_series,
                                                                                      random_state=random_state,
                                                                                      n_folds=n_folds, regr=regr, **kwargs)

        datasets_train_val_score.append(tuple(train_val_scores[best_index,:])) # Add the row related to that dataset
        datasets_best_params.append(params[best_index]) # Add the element related to that dataset
        datasets_test_score.append(test_score) # Add the row related to that dataset
        logger.log(logging.INFO, f'Dataset {i+1}/{len(dataset_list)}: best params: {params[best_index]}, test score: {test_score}')

    datasets_train_val_score = np.array(datasets_train_val_score) # Transform into numpy

    # Find the best index, i.e. the best dataset (more precisely, the best couple dataset-combination of hyperparameters
    # values)
    if regr:
        best_index = np.argmin(datasets_train_val_score,axis=0)[1]
    else:
        best_index = np.argmax(datasets_train_val_score,axis=0)[1]

    # Test score of the best couple dataset-combination of hyperparameters values
    test_score = datasets_test_score[best_index]
    logger.log(logging.INFO, f'Best couple dataset-combination of hyperparameters values: {datasets_best_params[best_index]}, test score: {test_score}')

    ax = None
    if(plot): # Make the plot
        if not xvalues: # Default values on the x axis
            xvalues = range(len(dataset_list))
        ax = _plot_TrainVal_values(xvalues,datasets_train_val_score,plot_train,xlabel,title,figsize, bar=True)

    return datasets_train_val_score, datasets_best_params, best_index, test_score, ax


def datasets_models_validation(dataset_list, model_paramGrid_list, scale_list=None, test_size=0.2, time_series=False,
                               random_state=123, n_folds=5, regr=True, plot=False, plot_train=False, xvalues=None,
                               xlabel="Datasets", title="Datasets validation", figsize=(6,6) ,verbose=False,
                               figsize_verbose=(6,6), logger=None, **kwargs):
    """
    Select the best dataset and the best model (i.e. select the best couple dataset-model).

    For each dataset in `dataset_list`, all the models in `model_paramGrid_list` are tested: each model is tested performing
    an exhaustive tuning of the specified hyperparameters. In fact, `model_paramGrid_list` also contains, for each model, the
    grid of the hyperparameters that have to be tested on that model (i.e. the grid which contains the values to test for
    each specified hyperparameter of the model).
    In other words, on each dataset the selection of the best model is performed: in fact, on each dataset, the function
    `models_validation` is applied. (See `models_validation`).
    In the end, the best couple dataset-model is selected.

    Despite the fact that a couple dataset-model is selected, the main viewpoint is focused with respect to the datasets.
    It's a validation focused on the datasets.
    In fact, first of all, for each dataset the model selection is performed: in this way the best model is selected
    and its relative score is associated with the dataset (i.e. it's the score of the dataset). (In other words, on each
    dataset the function `models_validation` is applied). Finally, after that, the best dataset is selected.
    It's a two-levels selection.

    This selection is made using the validation score (i.e. the best couple dataset-model is the one with best validation
    score).
    The validation score is computed by splitting each dataset into the training-test sets and then by applying the cross
    validation on the training set.
    Additionally, the training and test scores are also computed.

    Optionally, the validation scores of the datasets can be plotted, making a graphical visualization of the dataset
    selection. This is the 'main' plot.
    Moreover, still optionally, the 'secondary' plots can be done: for each dataset, the validation scores of the models are
    plotted, making a graphical visualization of the models selection on that dataset. (As the plot made by the
    `models_validation` function).

    Parameters
    ----------
    dataset_list: list
        List of couples, where each couple is a dataset.
            - The first element is X, the two-dimensional np.array containing the explanatory features of the dataset.
            - The second element is y, the mono dimensional np.array containing the response feature of the dataset.
    model_paramGrid_list: list
        List that specifies the models and the relative grid of hyperparameters to be tested.
        It's a list of triples (i.e. tuples), where each triple represents a model:
            - the first element is a string, which is a mnemonic name of that model;
            - the second element is the sklearn model;
            - the third element is the grid of hyperparameters to test for that model. It's a dictionary, with the same
              structure of parameter `param_grid` of the function `hyperparameters_validation`.
    scale_list: list or bool
        List of booleans, which has as many elements as the number of models to test (i.e. number of elements in the
        `model_paramGrid_list` list).
        This list indicates, for each different model, if the features in 'X' have to be scaled or not (for all the datasets).
        `scale_list` can be None or False: in this case the 'X' features aren't scaled for any model. `scale_list` can be
        True: in this case the 'X' features are scaled for all the models.
    test_size: float
        Decimal number between 0 and 1, which indicates the proportion of the test set (for each dataset).
    time_series: bool
        Indicates if the given datasets are time series dataset (i.e. datasets indexed by days).
        (This affects the computing of the validation score).
    random_state: int
        Used in the training-test splitting of the datasets.
    n_folds: int
        Indicates how many folds are made in order to compute the k-fold cross validation.
        (It's used only if `time_series` is False).
    regr: bool
        Indicates if it's either a regression or a classification problem.
    plot: bool
        Indicates whether to plot or not the validation score values of the datasets (i.e. this is the 'main' plot).
    plot_train: bool
        Indicates whether to plot also the training scores (both in the 'main' and 'secondary' plots).
    xvalues: list (in general, iterable)
        Values that have to be put in the x axis of the 'main' plot.
    xlabel: str
        Label of the x axis of the 'main' plot.
    title: str
        Title of the 'main' plot.
    figsize: tuple
        Two dimensions of the 'main' plot.
    verbose: bool
        If True, for each dataset the validation scores of the models are plotted (i.e. these are the 'secondary' plots).
        (See 'models_validation').
    figsize_verbose: tuple
        Two dimensions of the 'secondary' plots.

    Returns
    ----------
    datasets_train_val_score: np.array
        Two dimensional np.array, containing two columns: the first contains the training scores, the second the validation
        scores.
        It has as many rows as the number of datasets to test, i.e. as the number of elements in `dataset_list`.
    datasets_best_model: list
        List which has as many elements as the number of the datasets (i.e. number of elements in `dataset_list`). For
        each dataset, it contains the best model for that dataset.
        More precisely, it is a list of triple:
            - the first element is the index of `model_paramGrid_list` which indicates the best model;
            - the second element is the mnemonic name of the best model;
            - the third element is the best combination of hyperparameters values on that best model (i.e. it's a dictionary
              which has as keys the hyperparameters names and as values their associated values).
    best_index: int
        Index of `dataset_list` that indicates which is the best dataset.
    test_score: float
        Test score associated with the best couple dataset-model.
    axes: list
        List of the matplotlib Axes where the plots have been made.
        Firstly, the 'secondary' plots are put (if any). And, as last, the 'main' plot is put (if any).
        If no plot has been made, `axes` is an empty list.

    See also
    ----------
    models_validation: select the best model on the given dataset.

    Notes
    ----------
    - If `regr` is True, the validation scores are errors (MSE, i.e. Mean Squared Errors): this means that the best
      couple dataset-model is the one associated with the minimum validation score.
      Otherwise, the validation scores are accuracies: this means that the best couple is the one associated with the
      maximum validation score.
    - If `time_series` is False, the training-test splitting of each dataset is made randomly. In addition, the cross
      validation strategy performed is the classic k-fold cross validation: the number of folds is specified by `n_folds`.
      Otherwise, if `time_series` is True, the training-test sets are simply obtained by splitting each dataset into two
      contiguous parts. In addition, the cross validation strategy performed is the sklearn TimeSeriesSplit.
    """

    # numpy matrix (i.e. np.array) which has as many rows as the datasets, and it has the training and validation scores as
    # columns. At the beginning it is a list.
    datasets_train_val_score = []
    # List which contains, for each dataset, the best model. I.e. there is the triple index-model name-best combination of
    # hyperparameters values
    datasets_best_model = []
    # List which contains, for each dataset, its test score (associated to the best model)
    datasets_test_score = []
    # List of axes
    axes = []

    for i,dataset in enumerate(dataset_list): # Iterate through all the datasets

        X,y = dataset

        # Perform the models validation on the current dataset
        logger.log(logging.INFO, f'Performing the models validation on the dataset {i}')
        models_train_val_score, models_best_params, best_index, test_score, ax = models_validation(X, y,
                                                                                                   model_paramGrid_list,
                                                                                                   scale_list=scale_list,
                                                                                                   test_size=test_size,
                                                                                                   time_series=time_series,
                                                                                                   random_state=random_state,
                                                                                                   n_folds=n_folds,
                                                                                                   regr=regr, plot=verbose,
                                                                                                   plot_train=plot_train,
                                                                                                   xlabel="Models",
                                                                                                   title=("Dataset "+str(i)+
                                                                                                     " : models validation"),
                                                                                                   figsize=figsize_verbose, logger=logger, **kwargs)

        datasets_train_val_score.append(tuple(models_train_val_score[best_index,:])) # Add the row related to that dataset
        # Add the element related to that dataset
        datasets_best_model.append((best_index,model_paramGrid_list[best_index][0],models_best_params[best_index]))
        datasets_test_score.append(test_score) # Add the element related to that dataset
        logger.log(logging.INFO, f'\t The best model for the dataset {i} is {model_paramGrid_list[best_index][0]} with the following hyperparameters: {models_best_params[best_index]}')
        logger.log(logging.INFO, f'\t The test score for the dataset {i} is {test_score}')

        if ax:
            axes.append(ax)

    datasets_train_val_score = np.array(datasets_train_val_score) # Transform into numpy

    # Find the best index, i.e. the best dataset (more precisely, the best couple dataset-model)
    if regr:
        best_index = np.argmin(datasets_train_val_score,axis=0)[1]
    else:
        best_index = np.argmax(datasets_train_val_score,axis=0)[1]

    # Test score of the best couple dataset-model
    test_score = datasets_test_score[best_index]

    logger.log(logging.INFO, f'The best dataset is {best_index} with the following hyperparameters: {datasets_best_model[best_index][2]}')
    logger.log(logging.INFO, f'The test score of the best dataset is {test_score}')

    if(plot): # Make the plot
        if not xvalues: # Default values on the x axis
            xvalues = range(len(dataset_list))
        ax = _plot_TrainVal_values(xvalues,datasets_train_val_score,plot_train,xlabel,title,figsize, bar=True)
        axes.append(ax)

    return datasets_train_val_score, datasets_best_model, best_index, test_score, axes