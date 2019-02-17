"""
@InProceedings{Pang+Lee:04a,
  author =       {Bo Pang and Lillian Lee},
  title =        {A Sentimental Education: Sentiment Analysis Using Subjectivity Summarization Based on Minimum Cuts},
  booktitle =    "Proceedings of the ACL",
  year =         2004
}
"""
import math
import multiprocessing
# from argparse import ArgumentParser
from collections import Counter
# from tqdm import tqdm
from enum import Enum
from itertools import dropwhile
import re
# import argparse
import numpy as np
# import matplotlib.pyplot as plt
import spacy
import os
import time
from line_profiler import LineProfiler

lp = LineProfiler()
prf_dic = {}


class Model(Enum):
    Binary = "Binary"
    Unigram = "Unigram"


def prf(func):
    if func.__name__ not in prf_dic:
        prf_dic[func.__name__] = lp(func)
    return prf_dic[func.__name__]


class PerceptronClassifier:
    @prf
    def __init__(self, directory_path, mod=Model.Unigram, max_epoch=10):
        # processing data
        self.x_train_index, self.x_test_index = [], []
        self.x_train_value, self.x_test_value = [], []
        self.feature_dict = {'???': 0}
        self.feature_index = 1
        self.mod = mod
        print('------ current mode is %s ------' % self.mod)
        print('Run: data preparing...')
        self.__prepare_data(directory_path)
        self.y_train = [(lambda yy: 1 if yy < 800 else -1)(i) for i in range(1600)]
        self.y_test = [(lambda tt: 1 if tt < 200 else -1)(j) for j in range(400)]

        # training model
        print('Run: training...')
        self.sign = lambda xx: math.copysign(1, xx)
        self.w = np.zeros(self.feature_index)
        self.__train(max_epoch)

        # test model
        print('Run: testing...')
        self.__test()

    @prf
    def __prepare_data(self, directory_path):
        # pbar = tqdm(total=2000)
        count = 0
        nlp = spacy.load('en')

        neg_file_list, pos_file_list = os.listdir("%sneg/" % directory_path), os.listdir("%spos/" % directory_path)
        train_sample = pos_file_list[:800]
        train_sample.extend(neg_file_list[:800])
        test_sample = pos_file_list[800:]
        test_sample.extend(neg_file_list[800:])

        # pool = multiprocessing.Pool()
        # for x in range(80):
        #     pool.apply_async(self.hhh, train_sample)
        # pool.close()  # close函数表明不会再往进程池中加入新任务，一定要在join方法调用之前调用。
        # pool.join()

        # preparing training sample

        # for file in pos_file_list[:80]:
        #     path = '%spos/%s' % (directory_path, file)
        #     self.__doc_to_list_and_update_feature_dict(path)
        #     pbar.update(1)
        #
        # for file in neg_file_list[:80]:
        #     path = '%sneg/%s' % (directory_path, file)
        #     self.__doc_to_list_and_update_feature_dict(path)
        #     pbar.update(1)

        for file in train_sample:
            path = '%spos/%s' % (directory_path, file) if count < 800 else '%sneg/%s' % (directory_path, file)
            with open(path, 'r') as f:
                doc = nlp(re.sub("[^\w]", " ", f.read()))
            self.__doc_to_list_and_update_feature_dict(doc)
            count += 1
            # pbar.update(1)

        count = 0

        for file in test_sample:
            path = '%spos/%s' % (directory_path, file) if count < 200 else '%sneg/%s' % (directory_path, file)
            with open(path, 'r') as f:
                doc = nlp(re.sub("[^\w]", " ", f.read()))
            self.__prepare_testing_data(doc)
            count += 1
            # pbar.update(1)

        # # preparing testing sample
        # for file in pos_file_list[80:]:
        #     path = '%spos/%s' % (directory_path, file)
        #     self.__prepare_testing_data(path)
        #     pbar.update(1)
        #
        # for file in neg_file_list[80:]:
        #     path = '%sneg/%s' % (directory_path, file)
        #     self.__prepare_testing_data(path)
        #     pbar.update(1)

        # pbar.close()


        # file_list = os.listdir(directory_path + 'neg/')
        # file_list.extend(os.listdir(directory_path + 'pos/'))
        # print(file_list)

        # count_file, feature_index = 0, 0
        #
        # for file in file_list:
        #     # print(file)
        #     path = (directory_path + 'neg/' + file) if count_file < 100 else (directory_path + 'pos/' + file)
        #     with open(path, 'r') as f:
        #         doc = nlp(re.sub("[^\w]", " ", f.read()))
        #     # build the input feature matrix
        #     value_list, index_list = self.__doc_to_list_and_update_feature_dict(doc)
        #     if count_file < 80 or (99 < count_file < 180):
        #         self.x_train_value.append(value_list)
        #         self.x_train_index.append(index_list)
        #     else:
        #         self.x_test_value.append(value_list)
        #         self.x_test_index.append(index_list)
        #     count_file += 1
        #     pbar.update(1)
        # pbar.close()

    @prf
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

        self.x_train_index.append(index_list)
        self.x_train_value.append(value_list)

    @prf
    def __prepare_testing_data(self, doc):
        value_list, index_list = [1], [0]
        # nlp = spacy.load('en')
        # with open(path, 'r') as f:
        #     doc = nlp(re.sub("[^\w]", " ", f.read()))

        counter = Counter([token.lemma_ for token in doc if token.lemma_ in self.feature_dict])
        for lemma, value in counter.items():
            value_list.append(value)
            index_list.append(self.feature_dict[lemma])

        self.x_test_index.append(index_list)
        self.x_test_value.append(value_list)

    @prf
    def __train(self, max_epoch):
        np.random.seed(10)
        # pbar = tqdm(total=max_epoch)
        sample_list = [i for i in range(1600)]
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
            self.w = w_sum/(i+1)/1600
            error_rate_list.append(1 - self.__training_error())
            # pbar.update(1)

        # pbar.close()

        # print('Run: plotting training error...')
        # plt.figure(num=self.mod)
        # plt.plot(error_rate_list)
        # plt.title('Training Error for 10 epochs')
        # plt.show()

    @prf
    def __training_error(self):
        print('Calculating training error ....')
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

    def __test(self):
        prediction = []
        for i in range(400):
            features = self.x_test_index[i]
            x = self.x_test_value[i]
            w = self.w[features]
            prediction.append(self.__predict(x, w))
        # predictions = [self.__predict(self.x_test_value[i], self.w[self.x_test_index[i]]) for i in range(40)]
        accuracy_rate = self.__accuracy(prediction, self.y_test)
        print("accuracy rate is: {:.2f} \%".format(accuracy_rate * 100))
        return accuracy_rate

    def __predict(self, x, w):
        return self.sign(np.dot(x, w))

    @staticmethod
    def __accuracy(predictions, y):
        correct = np.count_nonzero(np.add(predictions, y))
        return correct / len(y)

    def __mod_name(self):
        if self.mod != 0:
            return 'binary' if self.mod == 1 else 'cut-off'
        else:
            return 'bag of words'






def show_profile_log():
    lp.print_stats()


@prf
def main():
    path = './review_polarity' + '/txt_sentoken/'
    # start = time.time()
    a = PerceptronClassifier(path)
    # print('time: ', time.time() - start)


# if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='Perceptron Classifier for Sentiment Analysis on Film Review.')
    # parser.add_argument('path', type=str, help='path to data directory')
    # args = parser.parse_args()

    # path = './review_polarity' + '/txt_sentoken/'
    #
    # start = time.time()
    # a = PerceptronClassifier(path)
    # print('time: ', time.time()-start)

    # start = time.time()
    # PerceptronClassifier(path, mod=1)
    # print('time: ', time.time() - start)
    #
    # start = time.time()
    # PerceptronClassifier(path, mod=2)
    # print('time: ', time.time() - start)
    # main()
    # show_profile_log()
    #
    # multy = multiprocessing.Pool()
    #

