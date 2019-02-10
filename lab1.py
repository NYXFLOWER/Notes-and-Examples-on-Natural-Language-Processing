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
from itertools import dropwhile
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt
import spacy
import os
import time


class PerceptronClassifier:
    def __init__(self, directory_path, mod=0, max_epoch=10):
        # processing data
        self.x_train_index, self.x_test_index = [], []
        self.x_train_value, self.x_test_value = [], []
        self.feature_dict = {'???': 0}
        self.feature_index = 1
        self.mod = mod
        print('------ current mode is %s ------' % self.mod)
        print('Run: data preparing...')
        self.__prepare_data(directory_path)
        self.y_train = [(lambda yy: 1 if yy < 80 else -1)(i) for i in range(160)]
        self.y_test = [(lambda tt: 1 if tt < 20 else -1)(j) for j in range(40)]

        # training model
        print('Run: training...')
        self.sign = lambda xx: math.copysign(1, xx)
        self.w = np.zeros(self.feature_index)
        self.__train(max_epoch)

        # test model
        print('Run: testing...')
        self.__test()

    def __prepare_data(self, directory_path):
        pbar = tqdm(total=200)
        nlp = spacy.load('en')
        file_list = os.listdir(directory_path + 'neg/')
        file_list.extend(os.listdir(directory_path + 'pos/'))
        # print(file_list)

        count_file, feature_index = 0, 0

        for file in file_list:
            # print(file)
            path = (directory_path + 'neg/' + file) if count_file < 100 else (directory_path + 'pos/' + file)
            with open(path, 'r') as f:
                doc = nlp(re.sub("[^\w]", " ", f.read()))
            # build the input feature matrix
            value_list, index_list = self.__doc_to_list_and_update_feature_dict(doc)
            if count_file < 80 or (99 < count_file < 180):
                self.x_train_value.append(value_list)
                self.x_train_index.append(index_list)
            else:
                self.x_test_value.append(value_list)
                self.x_test_index.append(index_list)
            count_file += 1
            pbar.update(1)
        pbar.close()

    def __doc_to_list_and_update_feature_dict(self, doc):
        value_list, index_list = [1], [0]
        if self.mod == 0:                               # bag of words
            counter = Counter([token.lemma_ for token in doc if token.lemma_ != '' and token.lemma_[0] != ' '])
            for lemma, value in counter.items():
                if lemma not in self.feature_dict:
                    self.feature_dict[lemma] = self.feature_index
                    self.feature_index += 1
                value_list.append(value)
                index_list.append(self.feature_dict[lemma])
            # lemma_to_value_index = {}
            # value_index = 1

            # for token in doc:
            #     lemma = token.lemma_
            #     if lemma[0] == ' ':
            #         continue
            #     if lemma not in self.feature_dict:
            #         self.feature_dict[lemma] = self.feature_index
            #         self.feature_index += 1
            #     if lemma not in lemma_to_value_index:
            #         lemma_to_value_index[lemma] = value_index
            #         value_index += 1
            #         index_list.append(self.feature_dict[lemma])
            #         value_list.append(1)
            #     else:
            #         value_list[lemma_to_value_index[lemma]] += 1


        elif self.mod == 1:                             # binary
            for token in doc:
                lemma = token.lemma_
                if lemma != ' ' and lemma not in self.feature_dict:
                    self.feature_dict[lemma] = self.feature_index
                    self.feature_index += 1
                index_list.append(self.feature_dict[lemma])
            value_list = [1 for k in range(self.feature_index)]

        else:
            counter = Counter([token.lemma_ for token in doc if token.lemma_[0] != ' '])
            for lemma, value in dropwhile(lambda key_count: key_count[1] < 3, counter.most_common()):
                del counter[lemma]
            for lemma, value in counter.items():
                if lemma not in self.feature_dict:
                    self.feature_dict[lemma] = self.feature_index
                    self.feature_index += 1
                value_list.append(value)
                index_list.append(self.feature_dict[lemma])

        return value_list, index_list

    def __train(self, max_epoch):
        np.random.seed(10)
        pbar = tqdm(total=max_epoch)
        sample_list = [i for i in range(160)]
        w_pre, w_sum = np.zeros(self.feature_index), np.zeros(self.feature_index)
        error_rate_list = []

        for i in range(max_epoch):
            np.random.shuffle(sample_list)
            for sample_index in sample_list:
                feature_index = self.x_train_index[sample_index]
                x = self.x_train_value[sample_index]
                y = self.y_train[sample_index]
                prediction = self.__predict(x, w_pre[feature_index])
                if prediction != y:
                    for j in range(len(x)):
                        w_pre[feature_index[j]] += y*x[j]
                w_sum += w_pre

            # calculate error rate for each epoch
            self.w = w_sum/(i+1)/160
            predictions = [self.__predict(self.x_train_value[j], self.w[self.x_train_index[j]]) for j in range(160)]
            error_rate_list.append(1 - self.__accuracy(predictions, self.y_train))
            pbar.update(1)
        pbar.close()

        print('Run: plotting training error...')
        plt.figure(num=self.mod)
        plt.plot([l for l in range(max_epoch)], error_rate_list)
        plt.title('Training Error for 10 epochs')
        plt.show()

    def __test(self):
        predictions = [self.__predict(self.x_test_value[i], self.w[self.x_test_index[i]]) for i in range(40)]
        accuracy_rate = self.__accuracy(predictions, self.y_test)
        print("accuracy rate is: {:.2f} \%".format(accuracy_rate * 100))

    def __predict(self, x, w):
        return self.sign(np.dot(x, w))

    def __accuracy(self, predictions, y):
        return np.count_nonzero(predictions + y) / len(y)

    def __mod_name(self):
        if self.mod != 0:
            return 'binary' if self.mod == 1 else 'cut-off'
        else:
            return 'bag of words'


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='Perceptron Classifier for Sentiment Analysis on Film Review.')
    # parser.add_argument('path', type=str, help='path to data directory')
    # args = parser.parse_args()

    path = './review_polarity' + '/txt_sentoken/'

    start = time.time()
    PerceptronClassifier(path)
    print('time: ', time.time()-start)

    # start = time.time()
    # PerceptronClassifier(path, mod=1)
    # print('time: ', time.time() - start)
    #
    # start = time.time()
    # PerceptronClassifier(path, mod=2)
    # print('time: ', time.time() - start)

