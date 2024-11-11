import numpy as np
from gensim.models import Word2Vec
from features import get_features_w2v
from classifier import search_C


def search_hyperparams(Xt_train, y_train, Xt_val, y_val):
    """Search the best values of hyper-parameters for Word2Vec as well as the
    regularisation parameter C for logistic regression, using the validation set.

    Args:
        Xt_train, Xt_val (list(list(list(str)))): Lists of (tokenised) documents (each
            represented as a list of tokenised sentences where a sentence is a
            list of tokens) for training and validation, respectively.
        y_train, y_val: Dense vectors (np.ndarray) of class labels for training
            and validation, respectively. Each element of the vector is either
            0 or 1.

    Returns:
        dict(str : union(int, float)): The best values of hyper-parameters.
    """
    # TODO: tune at least two of the many hyper-parameters of Word2Vec 
    #       (e.g. vector_size, window, negative, alpha, epochs, etc.) as well as
    #       the regularisation parameter C for logistic regression
    #       using the validation set.

    # The code below needs to be modified.
    best_params = dict()
    best_C = 1.0  # sklearn default
    best_acc = 0.0

    # hyperparameters to search
    vector_sizes = [100, 200, 300]
    windows = [2, 5, 10]
    epochs_list = [5, 10]

    # Iterate through all combinations of vector size, window, and epochs
    for vector_size in vector_sizes:
        for window in windows:
            for epochs in epochs_list:
                # Train Word2Vec model with current hyperparameters
                word_vectors = train_w2v(Xt_train, vector_size=vector_size, window=window, epochs=epochs)

                # Generate features for logistic regression
                X_train = get_features_w2v(Xt_train, word_vectors)
                X_val = get_features_w2v(Xt_val, word_vectors)

                # Search for the best C value
                C, val_acc = search_C(X_train, y_train, X_val, y_val, return_best_acc=True)
                
                # Track best performing hyperparameters
                if val_acc > best_acc:
                    best_acc = val_acc
                    best_params = {'C': C, 'vector_size': vector_size, 'window': window, 'epochs': epochs}
                    
    print(f'Best C value: {best_params["C"]}')
    print(f'Best vector size: {best_params["vector_size"]}')
    print(f'Best window size: {best_params["window"]}')
    print(f'Best epochs: {best_params["epochs"]}')
    print(f'Best accuracy on validation set: {best_acc}')
    
    return best_params


def train_w2v(Xt_train, vector_size=200, window=5, min_count=5, negative=10, epochs=3, seed=101, workers=10,
              compute_loss=False, **kwargs):
    """Train a Word2Vec model.

    Args:
        Xt_train (list(list(list(str)))): A list of (tokenised) documents (each
            represented as a list of tokenised sentences where a sentence is a
            list of tokens).
        See https://radimrehurek.com/gensim/models/word2vec.html#gensim.models.word2vec.Word2Vec
        for descriptions of the other arguments.

    Returns:
        gensim.models.keyedvectors.KeyedVectors: A mapping from words (string) to their embeddings
            (np.ndarray)
    """
    sentences_train = [sent for doc in Xt_train for sent in doc]

    # TODO: train the Word2Vec model
    print(f'Training word2vec using {len(sentences_train):,d} sentences ...')
 
    w2v_model = Word2Vec(sentences=sentences_train, vector_size=vector_size, window=window, min_count=min_count,
                         workers=workers, negative=negative, epochs=epochs, seed=seed, compute_loss=compute_loss)

    return w2v_model.wv

