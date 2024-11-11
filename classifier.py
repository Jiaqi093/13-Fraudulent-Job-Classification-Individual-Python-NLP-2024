import numpy as np
from scipy.sparse import issparse
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

def train_model(X_train, y_train, C):
    """Given a training dataset and a regularisation parameter
    return a logistic regression model fit to the dataset.

    Args:
        X_train: A (sparse or dense) matrix of features of documents.
            Each row is a document represented by its feature vector.
        y_train (np.ndarray): A vector of class labels, each element
            of the vector is either 0 or 1.
        C (float): Regularisation parameter C for LogisticRegression

    Returns:
        LogisticRegression: The trained logistic regression model.
    """
    # check the input
    assert X_train.shape[0] == y_train.shape[0]
    assert C > 0

    # train the logistic regression classifier
    model = LogisticRegression(C=C, max_iter=3000)
    model.fit(X_train, y_train)
    return model


def eval_model(X_test, y_test, model):
    """Given a model already fit to the training data, return the accuracy
        on the provided test data.

    Args:
        model (LogisticRegression): The trained logistic regression model
        X_test: A (sparse or dense) matrix of features of documents.
            Each row is a document represented by its feature vector.
        y_test (np.ndarray): A vector of class labels, each element of the 
            vector is either 0 or 1.

    Returns:
        float: The accuracy of the model on the provided data.
    """
    # check the input
    assert isinstance(model, LogisticRegression)
    assert X_test.shape[0] == y_test.shape[0]
    assert X_test.shape[1] == model.n_features_in_

    # test the logistic regression classifier and calculate the accuracy
    y_pred = model.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    
    # Calculate confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Calculate precision, recall, and F1-score for the positive class (fraudulent = 1)
    #precision = precision_score(y_test, y_pred, pos_label=1)
    #recall = recall_score(y_test, y_pred, pos_label=1)
    #f1 = f1_score(y_test, y_pred, pos_label=1)
    
    # calculate precision, recall, and F1-score for educational level
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    # Print all the metrics
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    return score


def search_C(X_train, y_train, X_val, y_val, return_best_acc=True):
    """Search the best value of hyper-parameter C using the validation set.

    Args:
        X_train, X_val: (Sparse or dense) matrices of document features for
            training and validation, respectively. Each row is a document
            represented by its feature vector.
        y_train, y_val: Dense vectors (np.ndarray) of class labels for training
            and validation, respectively. Each element of the vector is either
            0 or 1.
        return_best_acc (boolean): Optional. If True also return the best accuracy
            score on the validation set.

    Returns:
        float: The best C.
        float: Optional. The best accuracy score on the validation set.
    """
    param_grid = {'C': [0.001, 0.01, 0.1, 1, 5, 10, 50, 100, 200, 500, 750, 1000, 2000, 5000]}
    
    # Perform grid search using Logistic Regression
    grid = GridSearchCV(LogisticRegression(max_iter=3000), param_grid, cv=5)
    grid.fit(X_train, y_train)
    
    # /////////////////////////////////////////////////testing code/////////////////////////////////////////////////////////
    results = grid.cv_results_
    means = results['mean_test_score']  # Mean accuracy scores across the folds
    stds = results['std_test_score']    # Standard deviations of accuracy
    params = results['params']          # The parameter C values tested

    # Print the accuracy for each C value
    for mean, std, param in zip(means, stds, params):
        print(f"C={param['C']}: mean accuracy={mean:.4f} (+/-{std:.4f})")
    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
    # Get the best C value and accuracy on validation set
    best_C = grid.best_params_['C']
    best_acc = grid.best_score_
    
    return (best_C, best_acc) if return_best_acc else best_C

