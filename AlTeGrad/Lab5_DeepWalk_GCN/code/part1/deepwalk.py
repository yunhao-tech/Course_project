"""
Deep Learning on Graphs - ALTEGRAD - Nov 2022
"""

import numpy as np
import networkx as nx
from random import randint
from gensim.models import Word2Vec


############## Task 1
# Simulates a random walk of length "walk_length" starting from node "node"
def random_walk(G, node, walk_length):

    ##################
    # your code here #
    walk = [node]
    for i in range(walk_length-1):
        neignbours = list(G.neighbors(walk[i]))
        ind = randint(0, len(neignbours)-1)
        walk.append(neignbours[ind])
    ##################

    walk = [str(node) for node in walk]
    return walk


############## Task 2
# Runs "num_walks" random walks from each node
def generate_walks(G, num_walks, walk_length):
    walks = []
    
    ##################
    # your code here #
    for _ in range(num_walks):
        for node in G.nodes():
            walks.append(random_walk(G, node, walk_length))
    permuted_walks = np.random.permutation(walks)
    ##################

    return permuted_walks.tolist()


# Simulates walks and uses the Skipgram model to learn node representations
def deepwalk(G, num_walks, walk_length, n_dim):
    print("Generating walks")
    walks = generate_walks(G, num_walks, walk_length)

    print("Training word2vec")
    model = Word2Vec(vector_size=n_dim, window=8, min_count=0, sg=1, workers=8, hs=1)
    model.build_vocab(walks)
    model.train(walks, total_examples=model.corpus_count, epochs=5)

    return model
