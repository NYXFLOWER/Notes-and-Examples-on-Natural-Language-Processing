import multiprocessing
import os
import random
import re
import sys
from collections import Counter
from enum import Enum
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import spacy
from spacy.tokenizer import Tokenizer

DEFAULT_EPOCH = 10


# sign activation function
def sign(y):
    return -1 if y < 0 else 1


# split train, test set by index
def train_test_split(train_count=800, total_count=1000):
    # seed is set to a fixed number
    random.seed(180127760)
    # train set range [0-799, 1000-1799] in default
    train_set = list(range(train_count)) + list(range(total_count, total_count + train_count))
    # test set range [800-999, 1800-1999] in default
    test_set = list(range(train_count, total_count)) + list(range(total_count + train_count, 2 * total_count))

    # train set shuffling
    random.shuffle(train_set)

    return train_set, test_set


# read a text file by given path
def file_read(filepath):
    word_regex = re.compile(r"[\w]+[']*[\w]*")

    with open(filepath) as file:
        word_list = []
        for word in word_regex.findall(file.read().lower()):
            token = tokenizer(word)[0]
            # word stem to reduce the size of vector
            word_list.append(token.lemma_)

    return word_list


# model type enumeration
class Model(Enum):
    UNIGRAM = "Unigram"
    BIGRAM = "Bigram"
    TRIGRAM = "Trigram"


# perceptron classifier
class Classifier:

    # initialisation
    def __init__(self, word_lists, train_set, test_set, model_type=Model.UNIGRAM):

        self.accuracy_list = []
        self.word_dictionary_list = []
        self.word_index_dictionary = {}

        self.train_set = train_set
        self.test_set = test_set
        self.model_type = model_type

        self.word_extraction(word_lists)

    # extract unigram, bigram, or trigram from text
    def word_extraction(self, word_lists):

        for word_list in word_lists:

            words = []
            doc_word_dictionary = {}

            if self.model_type == Model.BIGRAM:
                # word is like "previous next" in bigram
                for index in range(len(word_list) - 1):
                    word = word_list[index] + " " + word_list[index + 1]
                    words.append(word)
            elif self.model_type == Model.TRIGRAM:
                # word is like "previous current next" in trigram
                for index in range(len(word_list) - 2):
                    word = word_list[index] + " " + word_list[index + 1] + " " + word_list[index + 2]
                    words.append(word)
            else:
                # word is still word in unigram
                words = word_list

            word_dictionary = Counter(words)
            for word in word_dictionary:
                if word not in self.word_index_dictionary:
                    # the index of a word is its order in the dictionary
                    self.word_index_dictionary[word] = len(self.word_index_dictionary)

                # the keys are word indices, values are word frequencies
                doc_word_dictionary[self.word_index_dictionary[word]] = word_dictionary[word]

            self.word_dictionary_list.append(doc_word_dictionary)

    def predict(self, doc, w):

        # x is word frequencies vector
        x = list(self.word_dictionary_list[doc].values())

        # sign(y_predict) = sign(w • x)
        sign_y_predict = sign(np.dot(w, x))

        # index lower than 1000 is positive reviews
        y = 1 if doc < 1000 else -1

        return sign_y_predict, y

    def train(self, epoch=DEFAULT_EPOCH):

        c = 0

        # initialise weights with small values
        w = np.empty(len(self.word_index_dictionary))
        w.fill(1e-10)
        w_sum = w
        w_mean = None

        for epoch_number in range(epoch):

            for doc in self.train_set:

                # indices of the words
                index_list = list(self.word_dictionary_list[doc].keys())
                # relevant weights
                w_rel = w[index_list]

                y_predict, y = self.predict(doc, w_rel)

                if y_predict != y:
                    if y == 1:
                        for word_index in index_list:
                            # w_c = w_c-1 + yΦ(x)
                            w[word_index] += self.word_dictionary_list[doc][word_index]
                    else:
                        for word_index in index_list:
                            # w_c = w_c-1 - yΦ(x)
                            w[word_index] -= self.word_dictionary_list[doc][word_index]

                c += 1
                w_sum = np.add(w_sum, w)

            # 1/c * sum(w)
            w_mean = w_sum / c
            self.test(w_mean)

        print("Final Accuracy (" + str(self.model_type.value) + "):", self.accuracy_list[-1] * 100, "%")
        print("Top 10 Features (" + str(self.model_type.value) + "):", self.top_ten_features(w_mean), "\n")

        return self.model_type, self.accuracy_list

    def test(self, w):

        error = 0

        for doc in self.test_set:

            # indices of the words
            index_list = list(self.word_dictionary_list[doc].keys())
            # relevant weights
            w_rel = w[index_list]

            y_predict, y = self.predict(doc, w_rel)

            if y_predict != y:
                error += 1

        # accuracy = (n - error) / n
        self.accuracy_list.append((len(self.test_set) - error) / len(self.test_set))

    def top_ten_features(self, w):

        # top 10 features' indices
        indices = w.argsort()[-10:][::-1]
        # words by indices
        keys = list(self.word_index_dictionary.keys())

        return [keys[index] for index in indices]


if __name__ == '__main__':

    nlp = spacy.load("en")
    tokenizer = spacy.tokenizer.Tokenizer(nlp.vocab)

    # process pool
    pool = multiprocessing.Pool()

    print("Importing Review Data...")
    # data paths formatting
    aaa = 'review_polarity'
    positive_review_paths = [path for path in glob(os.path.join(aaa + "/txt_sentoken/pos/", "*.txt"))]
    negative_review_paths = [path for path in glob(os.path.join(aaa + "/txt_sentoken/neg/", "*.txt"))]

    # read text files via multiprocessing
    text_list = pool.map(file_read, positive_review_paths)
    text_list += pool.map(file_read, negative_review_paths)
    print("Done\n")

    train, test = train_test_split()

    # three classifiers with different model types
    classifier_list = [
        Classifier(text_list, train, test, Model.UNIGRAM),
        Classifier(text_list, train, test, Model.BIGRAM),
        Classifier(text_list, train, test, Model.TRIGRAM)
    ]

    # asynchronous training
    async_results = []
    for classifier in classifier_list:
        print("Start", classifier.model_type.value, "Classifier Training")
        async_results.append(pool.apply_async(classifier.train))

    print()

    # wait for results
    pool.close()
    pool.join()

    for async_result in async_results:
        # plot learning progress according to model types
        model, accuracies = async_result.get()
        plt.plot(range(DEFAULT_EPOCH), accuracies, '-', label=model.value)

    plt.xticks(range(DEFAULT_EPOCH))
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Learning Progress")
    plt.legend()
    plt.show()
