"""
@InProceedings{Pang+Lee:04a,
  author =       {Bo Pang and Lillian Lee},
  title =        {A Sentimental Education: Sentiment Analysis Using Subjectivity Summarization Based on Minimum Cuts},
  booktitle =    "Proceedings of the ACL",
  year =         2004
}
"""
from argparse import ArgumentParser
from collections import Counter
import argparse
import numpy as np


class PerceptronClassifier:
    """ averaging weights, multiple passes, shuffling
    Keyword Arguments:
        - x_train / x_test: features - matrix - each row corresponds to a file, while each column corresponds to a word.
        - y_train / y_test: classes - a column vector - 1 for positive, 0 for negative
        - n_feature: number of features
        - max_iter: the max iteration for training
        - w_matrix =
    """
    def __init__(self, x_train, y_train, x_test, y_test, n_feature, max_iter=10):
        self._x = x_train
        self._y = y_train
        self._x_test = x_test
        self._y_test = y_test
        self._w_matrix = np.zeros(shape=(n_feature, max_iter))
        self.w = np.zeros(shape=(n_feature, 1))
        self.train(max_iter)

    def train(self, max_iter):
        for i in range(max_iter):



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perceptron Classifier for Sentiment Analysis on Film Review.')
    # parser.add_argument('path', type=str, help='path to data directory')
    # args = parser.parse_args()

    path = './review_polarity'
