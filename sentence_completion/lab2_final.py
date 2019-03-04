"""
Created by Hao Xu -- 27th February
This code runs and tests on MacOS 10.13.6

Command for run the code:
    $ python3 lab2.py news-corpus-500k.txt questions.txt

The data used in this code is from the following link.

"""
from scipy.sparse import coo_matrix, lil_matrix
import os
import sys
import numpy as np
import re
import time


class SentenceComplete:
    def __init__(self, corpus, coo_dict):
        """:param corpus: a list of words with <s> and </s>"""
        self.len_corpus = len(corpus)
        self.coordinate_dict = coo_dict
        self.count_lil = self.__construct_count_matrix(corpus)
        print("size of 157594*157594 'uint8' lil_matrix:     %5d bytes " % sys.getsizeof(lil_matrix))

    def __construct_count_matrix(self, corpus):
        # counter_unigram = Counter(corpus)
        # dimension = len(counter_unigram)

        # construct coordinate dictionary
        # keys = list(counter_unigram.keys())
        # values = list(i for i in range(dimension))

        # sparse matrix for unigram: diagonal
        # coo_uni = coo_matrix((list(counter_unigram.values()),
        #                       (values, values)),
        #                      shape=(dimension, dimension),
        #                      dtype=np.dtype('uint8'))

        # construct bigram counter (list of string pairs -> list of int pairs)
        # int_corpus = [self.coordinate_dict[word] for word in corpus]
        # counter_bigram = Counter(zip(int_corpus, int_corpus[1:]))

        # sparse matrix for bigram
        # rc = np.array(list(counter_bigram.keys()), dtype=str)
        # r = [self.coordinate_dict[a] for a in rc[:, 0]]
        # c = [self.coordinate_dict[a] for a in rc[:, 1]]

        # coo_bi = coo_matrix((list(counter_bigram.values()),
        #                      (r, c)),
        #                     shape=(dimension, dimension),
        #                     dtype=np.dtype('uint8'))

        # lil_bi = coo_matrix((dimension, dimension), dtype=np.dtype('uint8')).tolil()
        # for key, value in counter_bigram.items():
        #     lil_bi[key] = value

        # ----------------------------------------------------------#
        values = [1 for i in range(self.len_corpus)]
        # sparse matrix for unigram: diagonal
        coo_uni = coo_matrix((values, (corpus, corpus)),
                             shape=(self.len_corpus, self.len_corpus),
                             dtype=np.dtype('uint8'))

        # sparse matrix for bigram
        values.pop()
        coo_bi = coo_matrix((values, (corpus[:-1], corpus[1:])),
                            shape=(self.len_corpus, self.len_corpus),
                            dtype=np.dtype('uint8'))

        # ----------------------------------------------------------#
        coo = coo_uni + coo_bi
        print("size of 157594*157594 'uint8' coo_matrix : %5d bytes" % sys.getsizeof(coo))

        return coo.tolil()
        # return coo_uni.tolil() + lil_bi

    def test(self, questions, num_q=10):
        count_c1 = np.zeros(shape=(1, num_q), dtype=np.dtype('uint8'))
        count_c2 = np.zeros(shape=(1, num_q), dtype=np.dtype('uint8'))
        count_join_c1 = np.zeros(shape=(2, num_q), dtype=np.dtype('float64'))
        count_join_c2 = np.zeros(shape=(2, num_q), dtype=np.dtype('float32'))
        c1_list, c2_list = [], []

        print("size of 2*10 float64 nparray: %4d bytes" % sys.getsizeof(count_join_c1))
        print("size of 2*10 float32 nparray: %4d bytes" % sys.getsizeof(count_join_c2))

        for i in range(num_q):
            words = questions[i].split()
            blank_index = words.index("____")

            [c1, c2] = re.split("\/", words[-1])
            c1_list.append(c1), c2_list.append(c2)

            coo_c1, coo_c2 = self.coordinate_dict[c1], self.coordinate_dict[c2]
            coo_pre = self.coordinate_dict[words[blank_index - 1]]
            coo_next = self.coordinate_dict[words[blank_index + 1]]

            count_c1[0, i] = self.count_lil[coo_c1, coo_c1]
            count_c2[0, i] = self.count_lil[coo_c2, coo_c2]
            count_join_c1[:, i] = [self.count_lil[coo_pre, coo_c1], self.count_lil[coo_c1, coo_next]]
            count_join_c2[:, i] = [self.count_lil[coo_pre, coo_c2], self.count_lil[coo_c2, coo_next]]

        print("Join count: first row for candidate with the previous, second for the next")
        print("--> c1: ")
        print(count_join_c1)
        print("--> c2: ")
        print(count_join_c2)

        # Unigram
        print("Unigram: ")
        print("--> The first candidate:  ", ', '.join('{:6d}'.format(f) for f in count_c1.flatten()))
        print("--> The second candidate: ", ', '.join('{:6d}'.format(f) for f in count_c2.flatten()))

        # Bigram
        print("Bigram: ")
        score = count_join_c1[0, :] * count_join_c1[1, :] / count_c1
        print("--> The first candidate:  ", ', '.join('{:6.2f}'.format(f) for f in score.flatten()))
        score = count_join_c2[0, :] * count_join_c2[1, :] / count_c2
        print("--> The second candidate: ", ', '.join('{:6.2f}'.format(f) for f in score.flatten()))

        # Bigram with Smoothing
        print("Bigram with Smoothing: ")
        score = np.sum(np.log(count_join_c1 + 1), axis=0) - np.log(count_c1 + len(self.coordinate_dict))
        print("--> The first candidate:  ", ', '.join('{:6.2f}'.format(f) for f in score.flatten()))
        score = np.sum(np.log(count_join_c2 + 1), axis=0) - np.log(count_c2 + len(self.coordinate_dict))
        print("--> The second candidate: ", ', '.join('{:6.2f}'.format(f) for f in score.flatten()))

        # # Bigram with Smoothing
        # print("Bigram with Smoothing: ")
        # score = (count_join_c1[0, :] + 1) * (count_join_c1[1, :] + 1) / (np.array(count_c1) + len(self.coordinate_dict))
        # print("--> The first candidate:  ", ', '.join('{:6.5f}'.format(f) for f in score.flatten()))
        # score = (count_join_c2[0, :] + 1) * (count_join_c2[1, :] + 1) / (np.array(count_c2) + len(self.coordinate_dict))
        # print("--> The second candidate: ", ', '.join('{:6.5f}'.format(f) for f in score.flatten()))


# ################################################ #
#                   RUN FROM HERE                  #
# ################################################ #
if __name__ == '__main__':
    abs_path = os.path.abspath('.')
    # corpus_path = os.path.join(abs_path, sys.argv[1])
    # question_path = os.path.join(abs_path, sys.argv[2])

    t = time.time()
    # process corpus
    print("-------- Process Corpus and Questions --------")
    # path = '/Users/nyxfer/Documents/GitHub/nlp/sentence_completion/news-corpus-500k.txt'
    path = os.path.join(abs_path, sys.argv[1])
    with open(path, 'r') as f:
        s = f.readlines()
    corpus_list = []
    for line in s:      # add <s> and <\s> at the begin and end of each line
        corpus_list.append("<s>")
        corpus_list.extend(re.sub("[^\w]", ' ', line.lower()).split())     # convert to lower case
        corpus_list.append('</s>')

    dimension_word = set(corpus_list)
    dimension = len(dimension_word)
    print("Number of words in corpus: ", len(corpus_list))
    print("Number of unique words:    ", dimension)

    coordinate_dict = dict(zip(dimension_word, [i for i in range(dimension)]))
    corpus_list = [coordinate_dict[word] for word in corpus_list]  # int corpus

    # process questions
    # path = '/Users/nyxfer/Documents/GitHub/nlp/sentence_completion/questions.txt'
    path = os.path.join(abs_path, sys.argv[2])
    with open(path, 'r') as q:
        question_list = q.readlines()
    # answer_list = ['whether', 'through', 'piece', 'court', 'allowed',
    #           'check', 'hear', 'cereal', 'chews', 'sell']

    # train and test model
    print("-------- Train Language Models --------")
    sc = SentenceComplete(corpus_list, coordinate_dict)
    print("-------- Test Model Performance --------")
    sc.test(question_list)
    print("-------- Finished --------")
    print("Total time cost: %6.2f s" % (time.time() - t))



