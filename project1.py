# EECS 445 - Fall 2024
# Project 1 - project1.py

import numpy as np
import numpy.typing as npt
import pandas as pd
import yaml

from helper import *
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample

from sklearn.metrics import accuracy_score, precision_score, f1_score, roc_auc_score, average_precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve, auc

config = yaml.load(open("config.yaml"), Loader=yaml.SafeLoader)
seed = config["seed"]
np.random.seed(seed)


# Q1a
def generate_feature_vector(df: pd.DataFrame) -> dict[str, float]:
    """
    Reads a dataframe containing all measurements for a single patient
    within the first 48 hours of the ICU admission, and convert it into
    a feature vector.

    Args:
        df: dataframe with columns [Time, Variable, Value]

    Returns:
        a dictionary of format {feature_name: feature_value}
        for example, {'Age': 32, 'Gender': 0, 'max_HR': 84, ...}
    """
     # what are these two lines for?
    static_variables = config["static"]
    timeseries_variables = config["timeseries"]

    # TODO: 1) Replace unknown values with np.nan
    # NOTE: pd.DataFrame.replace() may be helpful here, refer to documentation for details
    df = df.replace(-1, np.nan)

    # Extract time-invariant and time-varying features (look into documentation for pd.DataFrame.iloc)
    static, timeseries = df.iloc[0:5], df.iloc[5:]

    feature_dict = {}
    # TODO: 2) extract raw values of time-invariant variables into feature dict
    for _, row in static.iterrows():
        feature_name = row["Variable"]
        feature_value = row["Value"]
        feature_dict[feature_name] = feature_value

    # TODO  3) extract max of time-varying variables into feature dict
    for variable in timeseries["Variable"].unique():
        # Get all measurements for this variable
        variable_data = timeseries[timeseries["Variable"] == variable]["Value"]
        max_value = variable_data.max() if not variable_data.empty else np.nan
        feature_dict[f"max_{variable}"] = max_value

    return feature_dict


# Q1b
def impute_missing_values(X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    For each feature column, impute missing values  (np.nan) with the
    population mean for that feature.

    Args:
        X: (N, d) matrix. X could contain missing values
    
    Returns:
        X: (N, d) matrix. X does not contain any missing values
    """
    # TODO: implement this function according to spec
    X_imputed = X.copy()

    # Iterate over each column (feature) of X
    for i in range(X.shape[1]):
        # Calculate the mean of the column ignoring np.nan values
        mean_value = np.nanmean(X[:, i])

        # Replace np.nan values in the column with the calculated mean
        X_imputed[np.isnan(X_imputed[:, i]), i] = mean_value
    return X_imputed


# Q1c
def normalize_feature_matrix(X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    For each feature column, normalize all values to range [0, 1].

    Args:
        X: (N, d) matrix.
    
    Returns:
        X: (N, d) matrix. Values are normalized per column.
    """
    # TODO: implement this function according to spec
    # NOTE: sklearn.preprocessing.MinMaxScaler may be helpful
    # Create the scaler with the range [0, 1]
    scaler = MinMaxScaler(feature_range=(0, 1))

    # Fit the scaler to the data and transform the matrix
    X_normalized = scaler.fit_transform(X)

    return X_normalized






# Helper function to calculate performance metrics
def calculate_metric(y_true, y_pred, metric="accuracy"):
    """
    Helper function to calculate performance metric.
    """
    if metric == "accuracy":
        return accuracy_score(y_true, y_pred)
    elif metric == "precision":
        return precision_score(y_true, y_pred, zero_division=0)
    elif metric == "f1-score":
        return f1_score(y_true, y_pred, zero_division=0)
    elif metric == "auroc":
        return roc_auc_score(y_true, y_pred)
    elif metric == "average_precision":
        return average_precision_score(y_true, y_pred)
    elif metric == "sensitivity":
        # cm = confusion_matrix(y_true, y_pred, labels=[-1, 1])
        # return cm[1, 1] / (cm[1, 1] + cm[1, 0])  # Sensitivity: TP / (TP + FN)
        return recall_score(y_true, y_pred)
    elif metric == "specificity":
        # cm = confusion_matrix(y_true, y_pred, labels=[-1, 1])
        # return cm[0, 0] / (cm[0, 0] + cm[0, 1])  # Specificity: TN / (TN + FP)
        tn, fp, _, _ = confusion_matrix(y_true, y_pred).ravel()
        return tn / (tn + fp)
    else:
        raise ValueError("Unsupported metric")


# currentlu not been used
def get_classifier(
    loss: str = "logistic",
    penalty: str | None = None,
    C: float = 1.0,
    class_weight: dict[int, float] | None = None,
    kernel: str = "rbf",
    gamma: float = 0.1,
) -> KernelRidge | LogisticRegression:
    """
    Return a classifier based on the given loss, penalty function
    and regularization parameter C.

    Args:
        loss: Specifies the loss function to use.
        penalty: The type of penalty for regularization (default: None).
        C: Regularization strength parameter (default: 1.0).
        class_weight: Weights associated with classes.
        kernel : Kernel type to be used in Kernel Ridge Regression. 
            Default is 'rbf'.
        gamma (float): Kernel coefficient (default: 0.1).
    Returns:
        A classifier based on the specified arguments.
    """
    # TODO (optional, but highly recommended): implement function based on docstring
    if loss == "logistic":
        return LogisticRegression(penalty=penalty, C=C, class_weight=class_weight, solver="liblinear")
    elif loss == "squared_error":  # for later use
        return KernelRidge(kernel=kernel, alpha=1.0/C, gamma=gamma)
    else:
        raise ValueError("Unsupported loss function")


def performance(
    clf_trained: KernelRidge | LogisticRegression,
    X: npt.NDArray[np.float64],
    y_true: npt.NDArray[np.int64],
    metric: str = "accuracy",
    bootstrap: bool=True
) -> tuple[np.float64, np.float64, np.float64] | np.float64:
    """
    Calculates the performance metric as evaluated on the true labels
    y_true versus the predicted scores from clf_trained and X, using 1,000 
    bootstrapped samples of the test set if bootstrap is set to True. Otherwise,
    returns single sample performance as specified by the user. Note: you may
    want to implement an additional helper function to reduce code redundancy.

    Args:
        clf_trained: a fitted instance of sklearn estimator
        X : (n,d) np.array containing features
        y_true: (n,) np.array containing true labels
        metric: string specifying the performance metric (default='accuracy'
                other options: 'precision', 'f1-score', 'auroc', 'average_precision', 
                'sensitivity', and 'specificity')
    Returns:
        if bootstrap is True: the median performance and the empirical 95% confidence interval in np.float64
        if bootstrap is False: peformance 
    """
    # TODO: Implement this function
    # This is an optional but VERY useful function to implement.
    # See the sklearn.metrics documentation for pointers on how to implement
    # the requested metrics.


    # bootstrap = False
    if not bootstrap:
        if isinstance(clf_trained, KernelRidge):
            # KernelRidge outputs continuous values, threshold at 0 for binary classification
            y_scores = clf_trained.predict(X)
            y_pred = np.where(y_scores >= 0, 1, -1)
            return calculate_metric(y_true, y_pred, metric)

        elif isinstance(clf_trained, LogisticRegression):
            if metric == "auroc" or metric == "average_precision":
                y_scores = clf_trained.decision_function(X)
                y_pred = y_scores  # For AUROC, we use the decision scores
            else:
                y_pred = clf_trained.predict(X)  # For other metrics, we use predicted labels
                # Single performance without bootstrapping
            return calculate_metric(y_true, y_pred, metric)
        else:
            raise ValueError("Unsupported classifier type")


    # Perform bootstrap sampling if bootstrap is True (this is for 2.1d)
    n_bootstraps = 1000
    bootstrapped_metrics = []

    for _ in range(n_bootstraps):
        # Resample the data
        X_boot, y_boot = resample(X, y_true)

        # Calculate metric for the bootstrapped sample
        if isinstance(clf_trained, KernelRidge):
            y_boot_scores = clf_trained.predict(X_boot)
            y_boot_pred = np.where(y_boot_scores >= 0, 1, -1)
        elif isinstance(clf_trained, LogisticRegression):
            if metric == "auroc" or metric == "average_precision":
                y_boot_scores = clf_trained.decision_function(X_boot)
                y_boot_pred = y_boot_scores
            else:
                y_boot_pred = clf_trained.predict(X_boot)

        boot_metric = calculate_metric(y_boot, y_boot_pred, metric)
        bootstrapped_metrics.append(boot_metric)

    # Convert bootstrapped metrics to a numpy array
    bootstrapped_metrics = np.array(bootstrapped_metrics)

    # Calculate median and 95% confidence intervals (2.5th percentile and 97.5th percentile)
    median_performance = np.median(bootstrapped_metrics)
    lower_ci = np.percentile(bootstrapped_metrics, 2.5)
    upper_ci = np.percentile(bootstrapped_metrics, 97.5)

    return median_performance, lower_ci, upper_ci


# Q2.1a
def cv_performance(
    clf: KernelRidge | LogisticRegression,
    X: npt.NDArray[np.float64],
    y: npt.NDArray[np.int64],
    metric: str = "accuracy",
    k: int = 5,
) -> tuple[float, float, float]:
    """
    Splits the data X and the labels y into k-folds and runs k-fold
    cross-validation: for each fold i in 1...k, trains a classifier on
    all the data except the ith fold, and tests on the ith fold.
    Calculates the k-fold cross-validation performance metric for classifier
    clf by averaging the performance across folds.
    
    Args:
        clf: an instance of a sklearn classifier
        X: (n,d) array of feature vectors, where n is the number of examples
           and d is the number of features
        y: (n,) vector of binary labels {1,-1}
        k: the number of folds (default=5)
        metric: the performance metric (default='accuracy'
             other options: 'precision', 'f1-score', 'auroc', 'average_precision',
             'sensitivity', and 'specificity')
    
    Returns:
        a tuple containing (mean, min, max) 'cross-validation' performance across the k folds
    """
    # TODO: Implement this function

    # NOTE: You may find the StratifiedKFold from sklearn.model_selection
    # to be useful
    skf = StratifiedKFold(n_splits=k, shuffle=False)

    # Store performance metrics for each fold
    performance_metrics = []

    # Perform k-fold cross-validation
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Fit the classifier
        clf.fit(X_train, y_train)

        # Use the performance function to calculate the performance
        fold_performance = performance(clf, X_test, y_test, metric=metric)
        performance_metrics.append(fold_performance)

    # TODO: Return the average, min,and max performance scores across all fold splits in a size 3 tuple.
    mean_performance = np.mean(performance_metrics)
    min_performance = np.min(performance_metrics)
    max_performance = np.max(performance_metrics)

    return (mean_performance, min_performance, max_performance)


# Q2.1b
def select_param_logreg(
    X: npt.NDArray[np.float64],
    y: npt.NDArray[np.int64],
    metric: str = "accuracy",
    k: int = 5,
    C_range: list[float] = [],
    penalties: list[str] = ["l2", "l1"],
) -> tuple[float, str]:
    """
    Sweeps different settings for the hyperparameter of a logistic regression,
    calculating the k-fold CV performance for each setting on X, y.
    
    Args:
        X: (n,d) array of feature vectors, where n is the number of examples
        and d is the number of features
        y: (n,) array of binary labels {1,-1}
        k: the number of folds (default=5)
        metric: the performance metric for which to optimize (default='accuracy',
             other options: 'precision', 'f1-score', 'auroc', 'average_precision', 'sensitivity',
             and 'specificity')
        C_range: an array with C values to be searched over
        penalties: a list of strings specifying the type of regularization penalties to be searched over
    
    Returns:
        The hyperparameters for a logistic regression model that maximizes the
        average k-fold CV performance.
    """
    # TODO: Implement this function
    # NOTE: You should be using your cv_performance function here
    # to evaluate the performance of each logistic regression classifier

    best_C = None
    best_penalty = None
    best_performance = -np.inf  # We want to maximize the performance

    # Iterate over all combinations of C and penalty
    # C_range = np.logspace(-3, 3, num=7)
    for C in C_range:
        for penalty in penalties:
            # Create a Logistic Regression model with current C and penalty
            clf = LogisticRegression(penalty=penalty, C=C, solver="liblinear", fit_intercept=False, random_state=seed)

            # Use the cv_performance function to get the mean performance
            mean_performance, _, _ = cv_performance(clf, X, y, metric=metric, k=k)

            # Update best parameters if we find a better performing model
            if mean_performance > best_performance:
                best_performance = mean_performance
                best_C = C
                best_penalty = penalty

    return best_C, best_penalty


# Q4c
def select_param_RBF(
    X: npt.NDArray[np.float64],
    y: npt.NDArray[np.int64],
    metric: str = "accuracy",
    k: int = 5,
    C_range: list[float] = [],
    gamma_range: list[float] = [],
) -> tuple[float, float]:
    """
    Sweeps different settings for the hyperparameter of a RBF Kernel Ridge Regression,
    calculating the k-fold CV performance for each setting on X, y.
    
    Args:
        X: (n,d) array of feature vectors, where n is the number of examples
        and d is the number of features
        y: (n,) array of binary labels {1,-1}
        k: the number of folds (default=5)
        metric: the performance metric (default='accuracy',
             other options: 'precision', 'f1-score', 'auroc', 'average_precision',
             'sensitivity', and 'specificity')
        C_range: an array with C values to be searched over
        gamma_range: an array with gamma values to be searched over
    
    Returns:
        The parameter value for a RBF Kernel Ridge Regression that maximizes the
        average k-fold CV performance.
    """
    print(f"RBF Kernel Ridge Regression Model Hyperparameter Selection based on {metric}:")
    # TODO: Implement this function acording to the docstring
    # NOTE: This function should be very similar in structure to your implementation of select_param_logreg()

    best_C = None
    best_gamma = None
    best_performance = -np.inf  # Want to maximize performance

    # Iterate over all combinations of C and gamma
    for C in C_range:
        for gamma in gamma_range:
            # Create the RBF Kernel Ridge Regression model with the current C and gamma
            clf = KernelRidge(kernel='rbf', alpha=1.0 / (2 * C), gamma=gamma)

            # Use the cv_performance function to get the mean performance
            mean_performance, _, _ = cv_performance(clf, X, y, metric=metric, k=k)

            # Update best parameters if we find a better performing model
            if mean_performance > best_performance:
                best_performance = mean_performance
                best_C = C
                best_gamma = gamma

    print(f"Best C: {best_C}, Best Gamma: {best_gamma}, Best {metric.capitalize()} Performance: {best_performance}")
    return best_C, best_gamma


# Q2.1e
def plot_weight(
    X: npt.NDArray[np.float64],
    y: npt.NDArray[np.int64],
    C_range: list[float],
    penalties: list[str],
) -> None:
    """
    The funcion takes training data X and labels y, plots the L0-norm
    (number of nonzero elements) of the coefficients learned by a classifier
    as a function of the C-values of the classifier, and saves the plot.
    Args:
        X: (n,d) array of feature vectors, where n is the number of examples
        and d is the number of features
        y: (n,) array of binary labels {1,-1}
    
    Returns:
        None
    """

    print("Plotting the number of nonzero entries of the parameter vector as a function of C")

    for penalty in penalties:
        # elements of norm0 should be the number of non-zero coefficients for a given setting of C
        norm0 = []
        for C in C_range:
            # TODO Initialize clf according to C and penalty
            clf = LogisticRegression(penalty=penalty, C=C, solver="liblinear", fit_intercept=False)

            # TODO Fit classifier with X and y
            clf.fit(X, y)

            # TODO: Extract learned coefficients/weights from clf into w
            # Note: Refer to sklearn.linear_model.LogisticRegression documentation
            # for attribute containing coefficients/weights of the clf object
            w = clf.coef_

            # TODO: Count number of nonzero coefficients/weights for setting of C
            #      and append count to norm0
            non_zero_count = np.sum(w != 0)
            norm0.append(non_zero_count)

        # This code will plot your L0-norm as a function of C
        plt.plot(C_range, norm0)
        plt.xscale("log")
    plt.legend([penalties[0], penalties[1]])
    plt.xlabel("Value of C")
    plt.ylabel("Norm of theta")

    # NOTE: plot will be saved in the current directory
    plt.savefig("L0_Norm.png", dpi=200)
    plt.close()





def main() -> None:
    print(f"Using Seed={seed}")
    # Read data
    # NOTE: READING IN THE DATA WILL NOT WORK UNTIL YOU HAVE FINISHED
    #       IMPLEMENTING generate_feature_vector, impute_missing_values AND normalize_feature_matrix
    # X_train, y_train, X_test, y_test, feature_names = get_train_test_split()

    metric_list = [
        "accuracy",
        "precision",
        "f1_score",
        "auroc",
        "average_precision",
        "sensitivity",
        "specificity",
    ]

    # TODO: Questions 1, 2, 3, 4
    # NOTE: It is highly recomended that you create functions for each
    #       sub-question/question to organize your code!

    # Helper Function to Test Multiple Classifiers!!
    def evaluate_classifier(clf, X_train, y_train, X_test, y_test, metrics):
        # Store the results in a table
        results = []

        # Fit the classifier
        clf.fit(X_train, y_train)

        # Evaluate the classifier for each metric
        for metric in metrics:
            median_perf, lower_ci, upper_ci = performance(clf, X_test, y_test, metric=metric, bootstrap=True)
            # Store the results for each metric
            results.append({
                "Metric": metric.capitalize(),
                "Median": f"{median_perf:.4f}",
                "95% CI Lower": f"{lower_ci:.4f}",
                "95% CI Upper": f"{upper_ci:.4f}"
            })

        # Create a DataFrame and print the table
        df = pd.DataFrame(results)
        print(df.to_string(index=False))


    # Question 1
    """ # Create a DataFrame for easier handling
    df_train = pd.DataFrame(X_train, columns=feature_names)

    # Calculate mean and interquartile range (IQR) for each feature
    summary_stats = []
    for feature in feature_names:
        mean_value = df_train[feature].mean()
        q25 = df_train[feature].quantile(0.25)
        q75 = df_train[feature].quantile(0.75)
        iqr_value = q75 - q25
        summary_stats.append([feature, mean_value, iqr_value])

    # Convert results into a DataFrame for display
    df_summary = pd.DataFrame(summary_stats, columns=["Feature Name", "Mean", "IQR"])

    # Display the result table
    print(df_summary) """



    # Question 2.1c
    # Performance measures you want to optimize for
    metrics = ["accuracy", "precision", "f1-score", "auroc", "average_precision", "sensitivity", "specificity"]

    """ # Store the results in a table
    results = []

    # Loop through each performance measure
    for metric in metrics:
        # Get the best C and penalty for the current metric using select_param_logreg
        best_C, best_penalty = select_param_logreg(X_train, y_train, metric=metric, k=5, C_range=np.logspace(-3, 3, num=7), penalties=["l2", "l1"])
        mean_perf, min_perf, max_perf = cv_performance(clf=LogisticRegression(), X=X_train, y=y_train, metric=metric, k=5)

        # Append results to the table
        results.append([metric, best_C, best_penalty, f"{mean_perf:.4f} ({min_perf:.4f}, {max_perf:.4f})"])

    # Print results in table format
    from tabulate import tabulate
    print(tabulate(results, headers=["Performance Measure", "C", "Penalty", "Mean (Min, Max) CV Performance"])) """



    # Question 2.1d
    """ clf_final = LogisticRegression(C=1.0, penalty="l1", solver="liblinear", fit_intercept=False)
    print("Results for best_in_1:")
    evaluate_classifier(clf_final, X_train, y_train, X_test, y_test, metrics) """



    # Question 2.1e
    """ plot_weight(X_train, y_train, C_range=np.logspace(-3, 3, num=7), penalties=["l2", "l1"]) """



    # Question 2.1f
    """ clf_l1 = LogisticRegression(C=1.0, penalty="l1", solver="liblinear", fit_intercept=False)
    clf_l1.fit(X_train, y_train)

    # Get the coefficient of the model (θ)
    theta = clf_l1.coef_[0]

    feature_names = np.array(feature_names)
    # Find the four most positive coefficients
    top_positive_indices = np.argsort(theta)[-4:][::-1]
    top_positive_coefficients = theta[top_positive_indices]
    top_positive_features = feature_names[top_positive_indices]

    # Find the four most negative coefficients
    top_negative_indices = np.argsort(theta)[:4]
    top_negative_coefficients = theta[top_negative_indices]
    top_negative_features = feature_names[top_negative_indices]

    # Output result
    positive_table = pd.DataFrame({
        "Feature": top_positive_features,
        "Coefficient": top_positive_coefficients
    })

    negative_table = pd.DataFrame({
        "Feature": top_negative_features,
        "Coefficient": top_negative_coefficients
    })

    print("Top 4 Positive Coefficients and Corresponding Features:")
    print(positive_table)

    print("\nTop 4 Negative Coefficients and Corresponding Features:")
    print(negative_table) """



    # Question 3.1b AND Question 3.2b
    """ clf_logreg_50 = LogisticRegression(
        penalty="l2", 
        C=1.0, 
        solver="liblinear", 
        class_weight={-1: 1, 1: 50},  # Wn = 1 and Wp = 50(3.1b)
        fit_intercept=False
    )
    print("Results for Wp = 50:")
    evaluate_classifier(clf_logreg_50, X_train, y_train, X_test, y_test, metrics)

    clf_logreg_100 = LogisticRegression(
        penalty="l2", 
        C=1.0, 
        solver="liblinear", 
        class_weight={-1: 1, 1: 100},  # Wn = 1 and Wp = 100(3.2b)
        fit_intercept=False
    )
    print("\nResults for Wp = 100:")
    evaluate_classifier(clf_logreg_100, X_train, y_train, X_test, y_test, metrics) """



    # Question 3.3a
    """ # Train the models with two sets of class weights
    clf1 = LogisticRegression(C=1.0, class_weight={-1: 1, 1: 1}, solver="liblinear")
    clf1.fit(X_train, y_train)

    clf2 = LogisticRegression(C=1.0, class_weight={-1: 1, 1: 5}, solver="liblinear")
    clf2.fit(X_train, y_train)

    # Get the decision scores for ROC curve
    y_scores_1 = clf1.decision_function(X_test)
    y_scores_2 = clf2.decision_function(X_test)

    # Compute ROC curve and AUC
    fpr1, tpr1, _ = roc_curve(y_test, y_scores_1)
    fpr2, tpr2, _ = roc_curve(y_test, y_scores_2)

    roc_auc1 = auc(fpr1, tpr1)
    roc_auc2 = auc(fpr2, tpr2)

    # Plot ROC curves
    plt.figure()
    plt.plot(fpr1, tpr1, label=f'Wn=1, Wp=1 (AUC = {roc_auc1:.2f})')
    plt.plot(fpr2, tpr2, label=f'Wn=1, Wp=5 (AUC = {roc_auc2:.2f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Different Class Weights')
    plt.legend(loc='lower right')
    plt.show()
    plt.close() """



    # Question 4b
    """ C=1.0
    clf_Logistic = LogisticRegression(penalty="l2", C=C, fit_intercept=False, random_state=seed)
    clf_KernelRidge = KernelRidge(kernel="linear", alpha=1.0/(2*C))
    evaluate_classifier(clf_Logistic, X_train, y_train, X_test, y_test, metrics)
    evaluate_classifier(clf_KernelRidge, X_train, y_train, X_test, y_test, metrics) """



    # Question 4d
    """ # Function to calculate cross-validation AUROC performance for different gamma values
    def report_rbf_cv_performance(X_train, y_train, C: float = 1.0, gamma_range: list[float] = [0.001, 0.01, 0.1, 1, 10, 100], k: int = 5):
        # Store the results in a table
        results = []

        # Iterate over the gamma values
        for gamma in gamma_range:
            # Create a Kernel Ridge Regression model with the RBF kernel
            clf = KernelRidge(kernel='rbf', alpha=1.0 / (2 * C), gamma=gamma). # kernel changed from "linear" to 'rbf' here

            # Perform cross-validation using AUROC as the metric
            mean_perf, min_perf, max_perf = cv_performance(clf, X_train, y_train, metric="auroc", k=k)

            # Store the results
            results.append({
                "Gamma": gamma,
                "Mean_AUROC": mean_perf,
                "Min_AUROC": min_perf,
                "Max_AUROC": max_perf
            })

        # Create a DataFrame for the results and print it
        df = pd.DataFrame(results)
        df["Performance"] = df.apply(lambda row: f"{row['Mean_AUROC']:.4f} ({row['Min_AUROC']:.4f}, {row['Max_AUROC']:.4f})", axis=1)
        print(df[["Gamma", "Performance"]].to_string(index=False))

    # Example call to the function
    report_rbf_cv_performance(X_train, y_train, C=1.0) """



    # Question 4e
    """ # Go through all the combinations of C and gamma
    C_range = [0.01, 0.1, 1.0, 10, 100]
    gamma_range = [0.01, 0.1, 1, 10]

    best_auroc = -np.inf
    best_C = None
    best_gamma = None

    # Store the results in a table
    results = []

    # We can no longer use the helper function 'evaluate_classifier' here because we have to compare the auroc score while
    # testing the model
    for C in C_range:
        for gamma in gamma_range:
            # Initialize Kernel Ridge model with RBF kernel
            clf = KernelRidge(alpha=1/(2*C), kernel='rbf', gamma=gamma)
            clf.fit(X_train, y_train)  # train model

            # Stores the metrics for each combination
            metrics_results = {}
            for metric in metrics:
                median_perf, lower_ci, upper_ci = performance(clf, X_test, y_test, metric=metric, bootstrap=True)
                metrics_results[metric] = (median_perf, lower_ci, upper_ci)

                # If AUROC is better, update the optimal parameters
                if metric == 'auroc' and median_perf > best_auroc:
                    best_auroc = median_perf
                    best_C = C
                    best_gamma = gamma

            # Prints the performance of the current combination
            print(f"C: {C}, Gamma: {gamma}")
            for metric, (median_perf, lower_ci, upper_ci) in metrics_results.items():
                print(f"{metric.capitalize()} - Median: {median_perf:.4f}, 95% CI: ({lower_ci:.4f}, {upper_ci:.4f})")

            # Store the results to the table
            result = {
                'C': C,
                'Gamma': gamma,
                'Performance': {metric: metrics_results[metric] for metric in metrics}
            }
            results.append(result)

    # Print optimal results
    print(f"Best AUROC achieved with C={best_C}, Gamma={best_gamma}, AUROC={best_auroc:.4f}")
    # The DataFrame of the final result
    df_results = pd.DataFrame(results) """





    # Read challenge data
    # TODO: Question 5: Apply a classifier to heldout features, and then use
    #       generate_challenge_labels to print the predicted labels
    X_challenge, y_challenge, X_heldout, feature_names = get_challenge_data()


    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.compose import ColumnTransformer
    from sklearn.model_selection import train_test_split
    from imblearn.over_sampling import SMOTE

    # 1. Handling Missing Values (Imputation with median)
    # 2. Scaling Numerical Features (Standardization)
    # 3. Handling Categorical Features (One-Hot Encoding)
    categorical_features = ['Gender', 'ICUType']
    numerical_features = [feature for feature in feature_names if feature not in categorical_features]

    # Pipeline for numerical features: impute missing values and standardize
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),  # Fill missing values with median
        ('scaler', StandardScaler())  # Standardize the numerical features
    ])

    # Pipeline for categorical features: one-hot encode
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),  # Handle missing categorical data
        ('onehot', OneHotEncoder(handle_unknown='ignore'))  # One-hot encode
    ])

    # Combine both numerical and categorical transformations
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )


    # Split the data into training and testing
    X_train_subset, X_test_subset, y_train_subset, y_test_subset = train_test_split(X_challenge, y_challenge, train_size=0.7, random_state=42)

    X_train_subset_df = pd.DataFrame(X_train_subset, columns=feature_names)
    X_test_df = pd.DataFrame(X_test_subset, columns=feature_names)

    X_train_scaled = preprocessor.fit_transform(X_train_subset_df)
    X_test_scaled = preprocessor.transform(X_test_df)

    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train_subset)

    # Perform a hyperparameter search
    param_grid = {'C': [0.01, 0.1, 1.0, 10, 100], 'penalty': ['l2']}
    grid_search = GridSearchCV(LogisticRegression(fit_intercept=False), param_grid, cv=5, scoring='roc_auc')
    grid_search.fit(X_train_resampled, y_train_resampled)

    # Output optimal parameters
    best_params = grid_search.best_params_
    print(f"Best parameters: {best_params}")

    clf = LogisticRegression(penalty="l2", C=0.1, fit_intercept=False)  # We get the penalty and C here from best_params we just have
    clf.fit(X_train_resampled, y_train_resampled)

    X_heldout_df = pd.DataFrame(X_heldout, columns=feature_names)
    # X_heldout_df['max_nan'] = X_heldout_df['max_nan'].fillna(0)  # For debug purpose
    X_heldout_scaled = preprocessor.transform(X_heldout_df)

    # Get binary prediction labels and risk scores
    y_label = clf.predict(X_heldout_scaled).astype(int)  # Converts the output to an integer
    y_score = clf.decision_function(X_heldout_scaled)
    generate_challenge_labels(y_label, y_score, "wuumaa")

    # 5. Print evaluation results and generate confusion matrix
    X_test_df = pd.DataFrame(X_test_subset, columns=feature_names)
    X_test_scaled = preprocessor.transform(X_test_df)

    y_cm = clf.predict(X_test_scaled).astype(int)
    cm = confusion_matrix(y_test_subset, y_cm)
    print("Confusion Matrix on Held-Out Set:\n", cm)

if __name__ == "__main__":
    main()
