import numpy as np


def print_dataset_distribution(y_train, y_val, y_test):
    """ Print the class distribution of the training, validation, and test sets.
        A helper function to visualise the class distribution of the dataset.

    Args:
        y_train: The class labels for the training set.
        y_val: The class labels for the validation set.
        y_test: The class labels for the test set.

    Returns: None
    """

    def print_distribution(y, set_name):
        class_counts = np.bincount(y)
        total = np.sum(class_counts)
        distribution = class_counts / total  # Get the distribution of each class
        print(f'{set_name} class distribution: {class_counts.tolist()}, distribution: {distribution.round(3).tolist()}')

    print_distribution(y_train, "Training set")
    print_distribution(y_val, "Validation set")
    print_distribution(y_test, "Test set")
