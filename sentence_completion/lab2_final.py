"""
Created by Hao Xu -- 27th February
This code runs and tests on MacOS 10.13.6

Command for run the code:
    $ python3 lab2.py news-corpus-500k.txt questions.txt

The data used in this code is from the following link.

"""

from collections import Counter
from scipy.sparse import coo_matrix, lil_matrix
import numpy as np
import re
import time


class SentenceComplete:
    def __init__(self, corpus):
        """:param corpus: a list of words with <s> and </s>"""

        self.coordinate_dict = {}
        self.count_lil = self.__construct_count_matrix(corpus)

    def __construct_count_matrix(self, corpus):
        counter_unigram = Counter(corpus)

        dimension = len(counter_unigram)

        # construct coordinate dictionary
        keys = list(counter_unigram.keys())
        values = list(i for i in range(dimension))
        self.coordinate_dict = dict(zip(keys, values))

        # sparse matrix for unigram: diagonal
        coo_uni = coo_matrix((list(counter_unigram.values()),
                              (values, values)),
                             shape=(dimension, dimension),
                             dtype=np.dtype('uint8'))

        # construct bigram counter (list of string pairs -> list of int pairs)
        int_corpus = [self.coordinate_dict[word] for word in corpus]
        counter_bigram = Counter(zip(int_corpus, int_corpus[1:]))

        # sparse matrix for bigram
        # rc = np.array(list(counter_bigram.keys()), dtype=str)
        # r = [self.coordinate_dict[a] for a in rc[:, 0]]
        # c = [self.coordinate_dict[a] for a in rc[:, 1]]

        # coo_bi = coo_matrix((list(counter_bigram.values()),
        #                      (r, c)),
        #                     shape=(dimension, dimension),
        #                     dtype=np.dtype('uint8'))

        lil_bi = coo_matrix((dimension, dimension), dtype=np.dtype('uint8')).tolil()
        for key, value in counter_bigram.items():
            lil_bi[key] = value

        return coo_uni.tolil() + lil_bi

    def test(self, questions, num_q=10):
        count_c1 = np.zeros(shape=(1, num_q), dtype=np.dtype('uint8'))
        count_c2 = np.zeros(shape=(1, num_q), dtype=np.dtype('uint8'))
        count_join_c1 = np.zeros(shape=(2, num_q), dtype=np.dtype('float64'))
        count_join_c2 = np.zeros(shape=(2, num_q), dtype=np.dtype('float64'))
        c1_list, c2_list = [], []

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

        print("Unique Words: ", len(self.coordinate_dict))
        format = ''
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
        print(count_join_c1, count_join_c2)
        score = np.sum(np.log(count_join_c1 + 1), axis=0) - np.log(np.array(count_c1) + len(self.coordinate_dict))
        print("--> The first candidate:  ", ', '.join('{:6.2f}'.format(f) for f in score.flatten()))
        score = np.sum(np.log(count_join_c2 + 1), axis=0) - np.log(np.array(count_c2) + len(self.coordinate_dict))
        print("--> The second candidate: ", ', '.join('{:6.2f}'.format(f) for f in score.flatten()))

        # Bigram with Smoothing
        print("Bigram with Smoothing: ")
        score = (count_join_c1[0, :] + 1) * (count_join_c1[1, :] + 1) / (np.array(count_c1) + len(self.coordinate_dict))
        print("--> The first candidate:  ", ', '.join('{:6.5f}'.format(f) for f in score.flatten()))
        score = (count_join_c2[0, :] + 1) * (count_join_c2[1, :] + 1) / (np.array(count_c2) + len(self.coordinate_dict))
        print("--> The second candidate: ", ', '.join('{:6.5f}'.format(f) for f in score.flatten()))


# ################################################ #
#                   RUN FROM HERE                  #
# ################################################ #
if __name__ == '__main__':
    t = time.time()
    # process corpus
    corpus_path = '/Users/nyxfer/Documents/GitHub/nlp/sentence_completion/news-corpus-500k.txt'
    with open(corpus_path, 'r') as f:
        s = f.readlines()
    corpus_list = []
    for line in s:      # add <s> and <\s> at the begin and end of each line
        corpus_list.append("<s>")
        corpus_list.extend(re.sub("[^\w]", ' ', line.lower()).split())     # convert to lower case
        corpus_list.append('</s>')
    print(time.time() - t)

    # process questions
    question_path = '/Users/nyxfer/Documents/GitHub/nlp/sentence_completion/questions.txt'
    with open(question_path, 'r') as q:
        question_list = q.readlines()
    # answer_list = ['whether', 'through', 'piece', 'court', 'allowed',
    #           'check', 'hear', 'cereal', 'chews', 'sell']

    # train and test model
    sc = SentenceComplete(corpus_list)
    print(time.time() - t)

    sc.test(question_list)
    print(time.time() - t)



