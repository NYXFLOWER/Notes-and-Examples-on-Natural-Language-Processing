"""
@InProceedings{Pang+Lee:04a,
  author =       {Bo Pang and Lillian Lee},
  title =        {A Sentimental Education: Sentiment Analysis Using Subjectivity Summarization Based on Minimum Cuts},
  booktitle =    "Proceedings of the ACL",
  year =         2004
}
"""
import math
from argparse import ArgumentParser
from collections import Counter
from tqdm import tqdm
import re
import argparse
import numpy as np
import spacy
import os


class PerceptronClassifier:
    """ averaging weights, multiple passes, shuffling
    Keyword Arguments:
        - x_train / x_test: features - matrix - each row corresponds to a review file, while each column corresponds to
                            a feature of review file.
        - y_train / y_test: classes - a list - 1 for positive, 0 for negative
        - n_feature: number of features
        - max_iter: the max iteration for training
        - w_matrix: each row corresponding to a sample
    """

    def __init__(self, x_train, y_train, max_iter=10):
        self.n_sample, self.n_feature = x_train.shape
        self._x = x_train
        # self._w_sum = np.zeros(shape=(1, self.n_feature))
        self._w_sum = np.zeros(self.n_feature)
        self.w = np.zeros(shape=(1, self.n_feature))
        self.y_test_pre_list = []
        self._train(max_iter, y_train)

    def _train(self, max_iter, y_train):
        np.random.seed(10)                                      # random seed for reproducible
        pbar = tqdm(total=max_iter)                             # processing bar
        # initial parameters
        index_list = np.arange(self.n_sample)
        index_w = 0
        accuracy_list = []

        for i in range(max_iter):                               # multiple passes
            np.random.shuffle(index_list)                       # shuffle training samples
            y_pre_list = []
            for ind in index_list:
                x_temp = self._x[ind, :]#.reshape(1, self.n_feature)
                y_temp = y_train[ind]  # int
                # w_temp = self._w_matrix[index_w - 1, :] if index_w != 0 else self._w_matrix[index_w, :]

                # y_pre = sign(np.dot(self._w_sum.reshape(1, self.n_feature), x_temp.reshape(self.n_feature, 1)))
                y_pre = sign(np.dot(self._w_sum, x_temp))
                y_pre_list.append(y_pre)

                self._w_sum += y_temp * x_temp
                print(self._w_sum)

                # self._w_matrix[index_w, :] = (w_temp + y_temp * x_temp) if y_pre != y_temp else w_temp
                # index_w += 1
            # accuracy for this iteration
            accuracy_list.append(self.accuracy(y_pre_list, y_train))
            # update processing bar
            pbar.update(10)
        print(accuracy_list)
        self.w = self._w_sum / max_iter / self.n_sample             # the average of all the weight vectors
        pbar.close()

    def prediction(self, x_test, y_test):
        y_pre = np.dot(self.w, x_test.T).flatten()
        y_pre_list = [sign(k) for k in y_pre]
        self.y_test_pre_list = y_pre_list
        return self.accuracy(y_pre_list, y_test)

    @staticmethod
    def accuracy(prediction, y):
        """ prediction and y are both list """
        temp = prediction + y
        return sum(1 for l in temp if l > 0)/len(y)


class NlpTools:
    def __init__(self):
        pass

    @staticmethod
    def read_from_directory(directory_path, num_train=800, tokenisation=True, lemmatisation=True, stoplist=True):
        """ :return a dict - value: list words for each document in the directory (can be preprocessed)
                           - key is the no. of document
                    train_dict - key < 800 is for negative, else positive
                    test_dict - key < 200 is for negative, else positive"""
        pbar = tqdm(total=2000)
        count_train, count_test, key, index = 0, 0, 0, 1            # index from 1 because the first feature of training sample is always 1
        train_dict, test_dict, feat_dict = {}, {}, {}
        string = ''
        nlp = spacy.load('en')
        # stoplist = ['a', 's', 'the', 'and', 'in', 'on', 'of', 'for', 'go', 'to', 'do', 'that', 'this', 'there', 'here']
        file_list = os.listdir(directory_path + 'neg/')
        file_list.extend(os.listdir(directory_path + 'pos/'))
        for key in range(2000):
            # for a file
            path = (directory_path + 'neg/' + file_list[key]) if key < 1000 else (directory_path + 'pos/' + file_list[key])
            with open(path, 'r') as f:
                doc = nlp(re.sub("[^\w]", " ", f.read()))
                lemma_list = [token.lemma_ for token in doc if token.lemma_[0] != ' ']  # and token.lemma_ not in stoplist]

                # for line in iter(f):
                #     string += line
                # string = f.read()
            # preprocessing data

            # change value of lemma_list from string to feature index
            for i in range(len(lemma_list)):
                lemma = lemma_list[i]
                # build/update feature dictionary
                if lemma not in feat_dict.keys():
                    feat_dict[lemma] = index
                    index += 1
                # change to index
                lemma_list[i] = feat_dict[lemma]

            # store to train dictionary
            if key < num_train or (999 < key < 1800):
                train_dict[count_train] = Counter(lemma_list)
                count_train += 1

            # store to test dictionary
            else:
                test_dict[count_test] = Counter(lemma_list)
                count_test += 1

            pbar.update(1)

        pbar.close()
        return train_dict, test_dict, feat_dict

    @staticmethod
    def generate_data(x_dict, n_feature):
        """ input is a dictionary, key is a list """
        data = np.zeros(shape=(len(x_dict.keys()), n_feature+1))
        data[:, 0] = 1
        for sample, sample_feature in x_dict.items():
            for feature, value in sample_feature.items():
                data[sample, feature] = value
        return data

def return_list(doc):


def pre_data(directory_path):
    count, feat_index, train_index, test_index = 0, 0, 1, 1
    feat_dict = {}
    x_train_index = [[]]
    x_train_value = [[]]
    x_test_index = [[]]
    x_test_value = [[]]

    pbar = tqdm(total=2000)
    nlp = spacy.load('en')
    file_list = os.listdir(directory_path + 'neg/')
    file_list.extend(os.listdir(directory_path + 'pos/'))

    for file in file_list:
        path = (directory_path + 'neg/' + file_list[count]) if count < 1000 else (directory_path + 'pos/' + file_list[count])
        with open(path, 'r') as f:
            doc = nlp(re.sub("[^\w]", " ", f.read()))


        for token in doc:
            lemma = token.lemma_
            if lemma == ' ':
                continue
            elif lemma not in feat_dict:
                feat_dict[lemma] = feat_index
                feat_index += 1



            lemma_list = [token.lemma_ for token in doc if token.lemma_[0] != ' ']  # and token.lemma_ not in stoplist]

        for i in range(len(lemma_list)):
            lemma = lemma_list[i]
            # build/update feature dictionary
            if lemma not in feat_dict.keys():
                feat_dict[lemma] = index
                index += 1
            # change to index
            lemma_list[i] = feat_dict[lemma]



    # stoplist = ['a', 's', 'the', 'and', 'in', 'on', 'of', 'for', 'go', 'to', 'do', 'that', 'this', 'there', 'here']

    for key in range(2000):



        # store to train dictionary
        if key < num_train or (999 < key < 1800):
            train_dict[count_train] = Counter(lemma_list)
            count_train += 1

        # store to test dictionary
        else:
            test_dict[count_test] = Counter(lemma_list)
            count_test += 1

        pbar.update(1)

    pbar.close()
    return train_dict, test_dict, feat_dict



sign = lambda xx: math.copysign(1, xx)
accuracy_format = '%.2f \%'


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='Perceptron Classifier for Sentiment Analysis on Film Review.')
    # parser.add_argument('path', type=str, help='path to data directory')
    # args = parser.parse_args()

    # neg_path = './review_polarity' + '/txt_sentoken/neg/'
    # pos_path = './review_polarity' + '/txt_sentoken/pos/'

    path = './review_polarity' + '/txt_sentoken/'
    # string = ''
    # for file in os.listdir(path):
    #     with open(path + file) as f:
    #         for line in iter(f):
    #             string += line
    #

    train_dict, test_dict, feat_dict = NlpTools.read_from_directory(path)

    n_feature = len(feat_dict)
    x_train = NlpTools.generate_data(train_dict, n_feature)
    x_test = NlpTools.generate_data(test_dict, n_feature)
    print(x_train)
    print(x_test)

    y_train = [(lambda yy: 1 if yy < 800 else -1)(i) for i in range(1600)]
    y_test = [(lambda tt: 1 if tt < 200 else -1)(j) for j in range(400)]
    # print(train_dict)
    # print(feat_dict)
    # nlp = spacy.load('en')
    # doc = nlp(re.sub("[^\w]", " ", string))
    # stoplist = ['a', 's', 'the', 'and', 'in', 'on', 'of', 'for', 'go', 'to', 'do', 'that', 'this', 'there', 'here']
    # # print(doc)
    # # token_list = [token for token in doc]
    # # print(token_list)
    # lemma_list =[token.lemma_ for token in doc if token.lemma_[0] != ' ' and token.lemma_ not in stoplist]
    # print(lemma_list)

    # x = np.array([[1, 2, 3, 4], [2, 3, 4, 5], [3, 7, 34, 2], [2, 3, 7, 3], [8, 5, 2, 22]])
    # y = [-1, 1, -1, 1, -1]
    # a = PerceptronClassifier(x_train, y_train)

    # the first feature type - bag of words


    # the second feature type - bag of words with stop list


    # the third feature type - bag of words with considerable words




