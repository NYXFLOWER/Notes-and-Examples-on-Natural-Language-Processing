"""
@InProceedings{Pang+Lee:04a,
  author =       {Bo Pang and Lillian Lee},
  title =        {A Sentimental Education: Sentiment Analysis Using Subjectivity Summarization Based on Minimum Cuts},
  booktitle =    "Proceedings of the ACL",
  year =         2004
}

This code runs and tests on MacOS 10.13.6
"""
import math
import multiprocessing
import os
import re
import time
from argparse import ArgumentParser
from collections import Counter
from enum import Enum
import matplotlib.pyplot as plt
import argparse
import numpy as np
import spacy
from line_profiler import LineProfiler

# lp = LineProfiler()
prf_dic = {}
nlp = spacy.load('en')


class Model(Enum):
    BagOfWords = "UNIGRAM"
    Bigram = "BIGRAM"
    Trigram = 'TRIGRAM'

    # @staticmethod
    # def mod_to_string():



# def prf(func):
#     if func.__name__ not in prf_dic:
#         prf_dic[func.__name__] = lp(func)
#     return prf_dic[func.__name__]
#
#
# def show_profile_log():
#     lp.print_stats()


def list_path(directory_path):
    neg_file_list, pos_file_list = os.listdir("%sneg/" % directory_path), os.listdir("%spos/" % directory_path)
    train_pos = pos_file_list[:800]
    train_neg = neg_file_list[:800]
    test_pos = pos_file_list[800:]
    test_neg = neg_file_list[800:]

    return [train_pos, train_neg, test_pos, test_neg], ['pos/', 'neg/', 'pos/', 'neg/']


def read_from_file(samples, directory_path, pn):
    doc_list = []
    # count = 0
    for file in samples:
        file_path = directory_path+pn+file
        with open(file_path, 'r') as f:
            doc_list.append(re.sub("[^\w]", " ", f.read()).split())
        # count += 1
        # if count % 20 == 0:
            # print(count)
    return doc_list


def read_from_file_wrap(args):
    return read_from_file(*args)


class PerceptronClassifier:
    # @prf
    def __init__(self, x_list, mod, max_epoch=15):
        # processing data
        self.x_train_index, self.x_test_index = [], []
        self.x_train_value, self.x_test_value = [], []
        self.feature_dict = {'???': 0}
        self.feature_index = 1
        self.mod = mod
        self.max_epoch = max_epoch

        self.sign = lambda xx: math.copysign(1, xx)

        # data generate for train
        self.generate_x_feature(x_list[0], x_list[1])
        self.y_train = [(lambda yy: 1 if yy < 800 else -1)(i) for i in range(1600)]
        self.w = np.zeros(self.feature_index)

        # data generate for test
        self.generate_test_x(x_list[2], x_list[3])
        self.y_test = [(lambda tt: 1 if tt < 200 else -1)(j) for j in range(400)]

        # train and test
        self.error_rate_list_train = []
        self.error_rate_list_test = []
        self.__train(max_epoch)
        self.__test()
        self.__print_top_ten_feature()

        # plot training and testing error and save to file
        self.__plot()

    # @prf
    def generate_x_feature(self, x_train_pos, x_train_neg):
        x_train = x_train_pos
        x_train.extend(x_train_neg)
        num_x = len(x_train)

        for i in range(num_x):
            # fit by feature type
            temp = x_train[i]
            if self.mod == Model.Bigram:
                num_bi = len(temp) - 1
                x = [''.join([temp[j], temp[j+1]]) for j in range(num_bi)]
            elif self.mod == Model.Trigram:
                num_tri = len(temp) - 2
                x = [''.join([temp[k], temp[k+1], temp[k+2]]) for k in range(num_tri)]
            else:
                x = temp

            # count frequency
            counter = Counter(x)
            value_list, index_list = self.__counter_to_lists(counter)

            self.x_train_index.append(index_list)
            self.x_train_value.append(value_list)

    # @prf
    def generate_test_x(self, x_test_pos, x_test_neg):
        x_test = x_test_pos
        x_test.extend(x_test_neg)
        num_x = len(x_test)

        for i in range(num_x):
            temp = x_test[i]
            if self.mod == Model.Bigram:
                num_bi = len(temp) - 1
                x = [''.join([temp[j], temp[j + 1]]) for j in range(num_bi)
                     if ''.join([temp[j], temp[j + 1]]) in self.feature_dict]
            elif self.mod == Model.Trigram:
                num_tri = len(temp) - 2
                x = [''.join([temp[k], temp[k + 1], temp[k + 2]]) for k in range(num_tri)
                     if ''.join([temp[k], temp[k + 1], temp[k + 2]]) in self.feature_dict]
            else:
                num_ui = len(temp)
                x = [temp[l] for l in range(num_ui) if temp[l] in self.feature_dict]

            counter = Counter(x)
            value_list, index_list = self.__counter_to_lists(counter)

            self.x_test_index.append(index_list)
            self.x_test_value.append(value_list)

    # @prf
    def __counter_to_lists(self, counter):
        value_list, index_list = [1], [0]
        for feature, frequency in counter.items():
            if feature not in self.feature_dict:
                self.feature_dict[feature] = self.feature_index
                self.feature_index += 1
            value_list.append(frequency)
            index_list.append(self.feature_dict[feature])
        return value_list, index_list

    # @prf
    def __train(self, max_epoch):
        np.random.seed(10)
        # pbar = tqdm(total=max_epoch)
        sample_list = [i for i in range(1600)]
        w_pre, w_sum = np.zeros(self.feature_index), np.zeros(self.feature_index)

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
            self.w = w_sum/(i+1)/1600
            accurcy = self.__training_error()
            self.error_rate_list_train.append(1 - accurcy)

            accurcy_test = self.__test()
            self.error_rate_list_test.append(1 - accurcy_test)
            # print("the accurcy is", accurcy)
            # pbar.update(1)

        # pbar.close()

        # print('Run: plotting training error...')

    # @prf
    def __training_error(self):
        # print('Calculating training error ....')
        prediction = []
        for i in range(1600):
            features = self.x_train_index[i]
            x = self.x_train_value[i]
            w = self.w[features]
            prediction.append(self.__predict(x, w))
        # predictions = [self.__predict(self.x_test_value[i], self.w[self.x_test_index[i]]) for i in range(40)]
        accuracy_rate = self.__accuracy(prediction, self.y_train)
        # print("accuracy rate is: {:.2f} \%".format(accuracy_rate * 100))
        return accuracy_rate

    # @prf
    def __test(self):
        prediction = []
        for i in range(400):
            features = self.x_test_index[i]
            x = self.x_test_value[i]
            w = self.w[features]
            prediction.append(self.__predict(x, w))
        # predictions = [self.__predict(self.x_test_value[i], self.w[self.x_test_index[i]]) for i in range(40)]
        accuracy_rate = self.__accuracy(prediction, self.y_test)
        print("Model ", self.mod, " - accuracy rate is: {:.2f} \%".format(accuracy_rate * 100))
        return accuracy_rate

    def __predict(self, x, w):
        return self.sign(np.dot(x, w))

    @staticmethod
    def __accuracy(predictions, y):
        correct = np.count_nonzero(np.add(predictions, y))
        return correct / len(y)

    # @prf
    def __print_top_ten_feature(self):
        features = list(self.feature_dict.keys())
        # positive
        ind_feature = np.argpartition(self.w, -10)[-10:]
        # top_feat = {features[i] : w[i] for i in ind_feature}
        print("The top ten positive features for ", self.mod, " :")
        for i in ind_feature:
            print("[", features[i], "]: ", "{:.2f}".format(self.w[i]))

        # negative
        # ind_feature = np.argpartition(self.w, 10)[:10]
        # print("The top ten negative features for ", self.mod, " :")
        # for i in ind_feature:
        #     print("[", features[i], "]: ", self.w[i])

    def __plot(self):
        xxx = [i+1 for i in range(self.max_epoch)]
        plt.figure()
        plt.plot(xxx, self.error_rate_list_train, 'r', label='training error')
        plt.plot(xxx, self.error_rate_list_test, 'y', label='testing error')
        plt.xticks(xxx)
        plt.ylim((0, 0.3))
        title = 'Training and Testing Error for ' + str(self.max_epoch) + ' epochs with ' + str(self.mod)
        plt.title(title)
        plt.grid()
        plt.legend()
        plt.savefig(title+'.pdf')
        # plt.show()


def init(res, mod):
    PerceptronClassifier(res, mod)


if __name__ == '__main__':
    t = time.time()
    path = './review_polarity/txt_sentoken/'
    path_list, pn_list = list_path(path)
    paths = [path for i in range(4)]

    # read from file and split text to string list
    pool = multiprocessing.Pool()
    x_res = pool.map(read_from_file_wrap, zip(path_list, paths, pn_list))
    pool.close()
    pool.join()
    print(time.time() - t)

    # x_res = [read_from_file(path_list[i], paths[i], pn_list[i]) for i in range(4)]

    t = time.time()
    # a = PerceptronClassifier(x_res, Model.Trigram)

    model = [Model.BagOfWords, Model.Bigram, Model.Trigram]
    # classifer = [init(x_res, model[i]) for i in range(3)]

    pool = multiprocessing.Pool()
    # [pool.apply_async(init, (x_res, model[i])) for i in range(3)]
    for i in range(3):
        # PerceptronClassifier(x_res, model[i])
        pool.apply_async(init, (x_res, model[i]))


    # res_list = [x_res for i in range(3)]
    # xxx = pool.map(init, zip(res_list, model))

    pool.close()
    pool.join()
    print(time.time() - t)

    # show_profile_log()
