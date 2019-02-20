"""
Created by Hao Xu -- 8th February
This code runs and tests on MacOS 10.13.6

Command for run the code:
    $ python lab1.py review_polarity

The data used in this code is from the following book.
@InProceedings{Pang+Lee:04a,
  author =       {Bo Pang and Lillian Lee},
  title =        {A Sentimental Education: Sentiment Analysis Using Subjectivity Summarization Based on Minimum Cuts},
  booktitle =    "Proceedings of the ACL",
  year =         2004
}
"""
import math
import multiprocessing
import os
import re
import time
import sys
from collections import Counter
from enum import Enum
import matplotlib.pyplot as plt
import numpy as np

# for each class (positive and negative)
NUM_TRAIN = 800
MUM_TEST = 200


# ################################################ #
#    read from files and pre-precessing texts      #
# ################################################ #
def list_path(directory_path):
    neg_file_list, pos_file_list = os.listdir("%sneg/" % directory_path), os.listdir("%spos/" % directory_path)
    train_pos = pos_file_list[:NUM_TRAIN]
    train_neg = neg_file_list[:NUM_TRAIN]
    test_pos = pos_file_list[NUM_TRAIN:]
    test_neg = neg_file_list[NUM_TRAIN:]

    return [train_pos, train_neg, test_pos, test_neg], ['pos/', 'neg/', 'pos/', 'neg/']


# use multi-processing to speed up
def read_from_file(samples, directory_path, pn):
    doc_list = []
    for file in samples:
        file_path = directory_path+pn+file
        with open(file_path, 'r') as f:
            doc_list.append(re.sub("[^\w]", " ", f.read()).split())
    return doc_list


# multi-processing tool
def read_from_file_wrap(args):
    return read_from_file(*args)


# ############################################### #
# Training and testing the proceptron classifiers #
# with the following three feature representation #
# ############################################### #
class Model(Enum):
    BagOfWords = "UNIGRAM"
    Bigram = "BIGRAM"
    Trigram = 'TRIGRAM'


class PerceptronClassifier:
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
        self.y_train = [(lambda yy: 1 if yy < NUM_TRAIN else -1)(i) for i in range(2*NUM_TRAIN)]
        self.w = np.zeros(self.feature_index)
        print("     -> Classifier with %s loaded" % str(mod))

        # data generate for test
        self.generate_test_x(x_list[2], x_list[3])
        self.y_test = [(lambda tt: 1 if tt < MUM_TEST else -1)(j) for j in range(2*MUM_TEST)]

        # train and test
        self.error_rate_list_train = []
        self.error_rate_list_test = []

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

            # append to list
            self.x_train_index.append(index_list)
            self.x_train_value.append(value_list)

    # only consider the features which have been already in the feature dictionary
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

    # function used in generate training and testing feature vectors
    def __counter_to_lists(self, counter):
        value_list, index_list = [1], [0]
        for feature, frequency in counter.items():
            if feature not in self.feature_dict:
                self.feature_dict[feature] = self.feature_index
                self.feature_index += 1
            value_list.append(frequency)
            index_list.append(self.feature_dict[feature])
        return value_list, index_list

    # train model
    def train(self, max_epoch):
        np.random.seed(10)                                          # random seed for reproducible
        sample_list = [i for i in range(2*NUM_TRAIN)]
        w_pre, w_sum = np.zeros(self.feature_index), np.zeros(self.feature_index)

        for i in range(max_epoch):                                  # multiple passes
            np.random.shuffle(sample_list)                          # shuffle training samples
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
            self.w = w_sum/(i+1)/2/NUM_TRAIN                         # the average of all the weight vectors
            accurcy = self.__training_error()
            self.error_rate_list_train.append(1 - accurcy)

            accurcy_test = self.__test()
            self.error_rate_list_test.append(1 - accurcy_test)

        self.__print_top_ten_feature()

    # used in train function
    def __training_error(self):
        prediction = []
        for i in range(2*NUM_TRAIN):
            features = self.x_train_index[i]
            x = self.x_train_value[i]
            w = self.w[features]
            prediction.append(self.__predict(x, w))
        accuracy_rate = self.__accuracy(prediction, self.y_train)
        return accuracy_rate

    # used in train function
    def __test(self):
        prediction = []
        for i in range(2*MUM_TEST):
            features = self.x_test_index[i]
            x = self.x_test_value[i]
            w = self.w[features]
            prediction.append(self.__predict(x, w))
        accuracy_rate = self.__accuracy(prediction, self.y_test)
        # print("Model ", self.mod, " - accuracy rate is: {:.2f}%".format(accuracy_rate * 100))
        return accuracy_rate

    # used in both train function and test function
    def __predict(self, x, w):
        return self.sign(np.dot(x, w))

    @staticmethod
    def __accuracy(predictions, y):
        correct = np.count_nonzero(np.add(predictions, y))
        return correct / len(y)

    # used in train function
    def __print_top_ten_feature(self):
        features = list(self.feature_dict.keys())
        ind_feature = np.argpartition(self.w, -10)[-10:]
        print("     The unsorted top ten positive features for ", self.mod, " :")
        for i in ind_feature:
            print("       -> [", features[i], "]: ", "{:.2f}".format(self.w[i]))

    # call at the end of the program
    def plot(self):
        xxx = [i+1 for i in range(self.max_epoch)]
        title = ''.join(['Training and Testing Error for ', str(self.max_epoch), ' epochs with ', str(self.mod)])
        plt.figure()
        plt.plot(xxx, self.error_rate_list_train, 'r', label='training error')
        plt.plot(xxx, self.error_rate_list_test, 'y', label='testing error')
        plt.xticks(xxx)
        plt.ylim((0, 0.3))
        plt.title(title)
        plt.grid()
        plt.legend()
        plt.savefig(title+'.jpg')
        plt.show()


# ################################################ #
#                   RUN FROM HERE                  #
# ################################################ #
if __name__ == '__main__':
    path = os.path.join(sys.argv[1] + "/txt_sentoken/")
    path_list, pn_list = list_path(path)
    pool = multiprocessing.Pool()

    # read from file and split text to string list
    print("--> Processing Data...")
    t1 = time.time()
    paths = [path for i in range(4)]
    x_res = pool.map(read_from_file_wrap, zip(path_list, paths, pn_list))
    print("Finished... Time cost: ", "{:.2f}".format(time.time() - t1), "\n")

    # initial classifiers
    print("--> Training and Testing...")
    t2 = time.time()
    model_list = [Model.BagOfWords, Model.Bigram, Model.Trigram]
    classifier_list = [PerceptronClassifier(x_res, mod) for mod in model_list]
    print()
    # training
    [pool.apply_async(classifier.train(15)) for classifier in classifier_list]

    # close multiple processing and wait for merging
    pool.close()
    pool.join()

    print("Finished ALL... Time cost: ", "{:.2f}".format(time.time() - t2))

    # plot training and testing error for each classifier
    for classifier in classifier_list:
        classifier.plot()
