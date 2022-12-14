"""
Learning on Sets / Learning with Proteins - ALTEGRAD - Dec 2022
"""

import numpy as np


def create_train_dataset():
    n_train = 100000
    max_train_card = 10

    ############## Task 1
    
    ##################
    # your code here #
    X_train = np.zeros((n_train, max_train_card))
    y_train = np.zeros(n_train)
    for i in range(n_train):
        card = np.random.randint(1, max_train_card+1)
        X_train[i, -card:] = np.random.randint(1, max_train_card, size=card)
        y_train[i] = np.sum(X_train[i,:])
    
    print(f"First training sample: {X_train[0,:]}")
    print(f"Target of First training sample: {y_train[0]}")


    ##################

    return X_train, y_train


def create_test_dataset():
    
    ############## Task 2
    
    ##################
    # your code here #
    min_test_card = 5
    max_test_card = 100
    step_test_card = 5
    n_test = 200000
    cards = range(min_test_card, max_test_card+1, step_test_card)
    n_samples_per_card = n_test // len(cards)
    
    X_test = list()
    y_test = list()
    for card in cards:
        X = np.random.randint(1, 11, size=(n_samples_per_card, card))
        y = np.sum(X, axis=1)
        
        X_test.append(X)
        y_test.append(y)
    
    ##################

    return X_test, y_test