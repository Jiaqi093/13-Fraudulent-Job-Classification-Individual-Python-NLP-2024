import os
import nltk
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from helper import print_dataset_distribution
from sklearn.model_selection import StratifiedShuffleSplit
from features import (
    tokenise_text,
    get_features_tfidf,
    get_features_w2v,
)
from word2vec import (
    train_w2v,
    search_hyperparams,
)
from classifier import (
    train_model,
    eval_model,
    search_C,
)

# This is an arbitrary number chosen as the random seed.
random_state = 4650_2034


def prepare_dataset(filename):
    """Prepare the training/validation/test dataset.

    Args:
        filename (str): The name of file from which data will be loaded.

    Returns:
        Xr_train (iterable(str)): Documents in the training set, each 
            represented as a string.
        y_train (np.ndarray): A vector of class labels for documents in 
            the training set, each element of the vector is either 0 or 1.
        Xr_val (iterable(str)): Documents in the validation set, each 
            represented as a string.
        y_val (np.ndarray): A vector of class labels for documents in 
            the validation set, each element of the vector is either 0 or 1.
        Xr_test (iterable(str)): Documents in the test set, each 
            represented as a string.
        y_test (np.ndarray): A vector of class labels for documents in 
            the test set, each element of the vector is either 0 or 1.
    """
    print('Preparing train/val/test dataset ...')
    # load raw data
    df = pd.read_csv(filename)

    # Replace NaN values with an empty string
    df = df.fillna('')

    df = df[df['fraudulent'] == 0]
    df = df[df['required_education'].isin(["Master's Degree", "Bachelor's Degree", "High School or equivalent"])]
    
    # shuffle the rows
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # extract relevant columns to construct X and y

    # TODO (Optional): 
    # Currently we use only the "description" column as X. You can consider 
    # concatenating multiple columns as X.
    Xr = df['description']
    y = df['required_education']
    #y = df['fraudulent']

    # Encode the target label
    le = LabelEncoder()
    y = le.fit_transform(y)

    # Split into training, validation, and test sets
    train_frac, val_frac, test_frac = 0.7, 0.1, 0.2
    # Variable initializing
    Xr_temp, y_temp, Xr_train, Xr_val,  Xr_test, y_train, y_val, y_test = None, None, None, None, None, None, None, None

    # NOTE: The split is done in two steps to ensure that the distribution of the dataset is maintained
    # Initialize StratifiedShuffleSplit with the desired proportions
    split = StratifiedShuffleSplit(n_splits=1, test_size=(val_frac + test_frac),
                                   random_state=random_state)
    # Split the dataset into training and temp sets (temp will be further split into val and test)
    for train_index, temp_index in split.split(Xr, y):
        Xr_train, Xr_temp = Xr.iloc[train_index], Xr.iloc[temp_index]
        y_train, y_temp = y[train_index], y[temp_index]

    # Further split the temp set into validation and test sets
    split_temp = StratifiedShuffleSplit(n_splits=1, test_size=(test_frac / (val_frac + test_frac)),
                                        random_state=random_state)
    # Split the temp set into validation and test sets
    for val_index, test_index in split_temp.split(Xr_temp, y_temp):
        Xr_val, Xr_test = Xr_temp.iloc[val_index], Xr_temp.iloc[test_index]
        y_val, y_test = y_temp[val_index], y_temp[test_index]

    # Print the distribution of the dataset based on feature (Y)
    print_dataset_distribution(y_train, y_val, y_test)

    # Turn X into lists
    Xr_train = Xr_train.tolist()
    Xr_val = Xr_val.tolist()
    Xr_test = Xr_test.tolist()

    return Xr_train, y_train, Xr_val, y_val, Xr_test, y_test


def analyse_classification_tfidf(Xr_train, y_train, Xr_val, y_val, Xr_test, y_test):
    """Analyse classification using TF-IDF features.

    Args:
        Xr_train (iterable(str)): Documents in the training set, each 
            represented as a string.
        y_train (np.ndarray): A vector of class labels for documents in 
            the training set.
        Xr_val (iterable(str)): Documents in the validation set, each 
            represented as a string.
        y_val (np.ndarray): A vector of class labels for documents in 
            the validation set.
        Xr_test (iterable(str)): Documents in the test set, each 
            represented as a string.
        y_test (np.ndarray): A vector of class labels for documents in 
            the test set.

    Returns:
        float: The accuracy of the classification classifier on the test set.
    """
    # generate TF-IDF features for texts in the training and validation sets 
    X_train, X_val = get_features_tfidf(Xr_train, Xr_val)

    # search for the best C value
    C, best_acc = search_C(X_train, y_train, X_val, y_val)
    print(f'Best C value: {C}')
    print(f'Best accuracy on validation set: {best_acc}')
    
    print('Analysing classification (TF-IDF) ...')

    # re-train the classifier using the training set concatenated with the
    # validation set and the best C value
    X_train_val, X_test = get_features_tfidf(Xr_train + Xr_val, Xr_test)
    y_train_val = np.concatenate([y_train, y_val], axis=-1)
    model = train_model(X_train_val, y_train_val, C)

    # evaluate performance on the test set
    acc = eval_model(X_test, y_test, model)
    return acc


def analyse_classification_w2v(Xr_train, y_train, Xr_val, y_val, Xr_test, y_test):
    """Analyse classification using aggregated word2vec word vectors.

    Args:
        Xr_train (iterable(str)): Documents in the training set, each 
            represented as a string.
        y_train (np.ndarray): A vector of class labels for documents in 
            the training set.
        Xr_val (iterable(str)): Documents in the validation set, each 
            represented as a string.
        y_val (np.ndarray): A vector of class labels for documents in 
            the validation set.
        Xr_test (iterable(str)): Documents in the test set, each 
            represented as a string.
        y_test (np.ndarray): A vector of class labels for documents in 
            the test set.
        word2vec_model (Word2VecModel): A trained word2vec model.

    Returns:
        float: The accuracy of the classification classifier on the test set.
    """
    # prepare data
    get_sentences = lambda text: [tokenise_text(sent) for sent in nltk.tokenize.sent_tokenize(text)]
    Xt_train = [get_sentences(xr) for xr in tqdm(Xr_train)]
    Xt_val = [get_sentences(xr) for xr in tqdm(Xr_val)]
    Xt_test = [get_sentences(xr) for xr in tqdm(Xr_test)]

    # tune hyper-parameters
    best_params = search_hyperparams(Xt_train, y_train, Xt_val, y_val)
    assert 'C' in best_params
    best_C = best_params['C']
    del best_params['C']
    word_vectors = train_w2v(Xt_train + Xt_val, **best_params)
    
    # re-train the classifier using the training set concatenated with the
    # validation set
    X_train_val = get_features_w2v(Xt_train + Xt_val, word_vectors)
    y_train_val = np.concatenate([y_train, y_val], axis=-1)
    X_test = get_features_w2v(Xt_test, word_vectors)
    print('Analysing classification (word2vec) ...')
    model = train_model(X_train_val, y_train_val, best_C)

    # evaluate performance on the test set
    acc = eval_model(X_test, y_test, model)
    return acc


if __name__ == '__main__':
    # Get the data file path for the dataset
    # TODO: replace the name of the file below with the data file
    # you have downloaded from Kaggle or with a data file that you
    # have preprocessed.
    data_file = os.path.join("data", "fake_job_postings.csv")

    # split the dataset into training, validation, and test sets for
    # the education classification task
    Xr_train, y_train, Xr_val, y_val, Xr_test, y_test = prepare_dataset(filename=data_file)

    # uncomment to perform classification on the test set using TF-IDF features
    #acc = analyse_classification_tfidf(Xr_train, y_train, Xr_val, y_val, Xr_test, y_test)
    #print(f'Accuracy on test set (TF-IDF): {acc:.4f}')

    # uncomment to perform classification on the test set using aggregated word vectors
    acc = analyse_classification_w2v(Xr_train, y_train, Xr_val, y_val, Xr_test, y_test)
    print(f'Accuracy on test set (word2vec): {acc:.4f}')
