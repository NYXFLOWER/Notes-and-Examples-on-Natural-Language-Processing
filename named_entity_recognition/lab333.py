"""
Created by Hao Xu -- 10th Mar
This code runs and tests on MacOS 10.13.6

Command for run the code:
    $ python3 lab3.py train.txt test.txt

The data used in this code is from the following link.

"""
import os
from collections import Counter
from itertools import product, chain, dropwhile
import numpy as np
from scipy.sparse import csr_matrix


def __load_dataset_sents(file_path, as_zip=True, to_idx=False, token_vocab=None, target_vocab=None):
    targets = []
    inputs = []
    zip_inps = []
    with open(file_path) as f:
        for line in f:
            sent, tags = line.split('\t')
            words = [token_vocab[w.strip()] if to_idx else w.strip() for w in sent.split()]
            ner_tags = [target_vocab[w.strip()] if to_idx else w.strip() for w in tags.split()]
            inputs.append(words)
            targets.append(ner_tags)
            zip_inps.append(list(zip(words, ner_tags)))
    return zip_inps if as_zip else (inputs, targets)


def __preprocess_data(label_index_dict, train_data_list):
    """ Function for indexing word in corpus and reconstruct training dataset. """
    word_index_dict = {}
    train_list = []
    temp_list = []
    w_index = 0

    for sent in train_data_list:
        temp_list = []
        for (w, l) in sent:
            if w not in word_index_dict:
                word_index_dict[w] = w_index
                w_index += 1
            temp_list.append((word_index_dict[w], label_index_dict[l]))
        train_list.append(temp_list)

    return word_index_dict, train_list


class CandidateLabelSpace:
    def __init__(self, max_dim, num_label=5):
        self.label_dim_dict = {}
        self.max_dim = max_dim
        self.generate_candidate_label_sequence(num_label)

    def generate_candidate_label_sequence(self, num_label):
        labels = [i for i in range(num_label)]
        for i in range(self.max_dim):
            self.label_dim_dict[i] = list(product(labels, repeat=(i+1)))


class StructuredProceptron:
    def __init__(self, corpus):
        self.corpus = corpus
        self.label_space = self.__construct_candidate_label()

        # model 1
        self.cscl_space = self.__construct_cscl()    # Current Sample-Current Label Feature Space
        self.coo_1 = dict(zip(self.cscl_space.keys(), list(range(len(self.cscl_space)))))

        # model 2
        self.cscl_plcl_space = self.__construct_plcl() | self.cscl_space    # Previous Label-Current Label Feature Space
        self.coo_2 = dict(zip(self.cscl_plcl_space.keys(), list(range(len(self.cscl_plcl_space)))))

        # model 3
        self.cscl_plcl_pscs_space = self.__construct_pscs() | self.cscl_plcl_space    # Previous Sample-Current Sample Feature Space
        self.coo_3 = dict(zip(self.cscl_plcl_pscs_space.keys(), list(range(len(self.cscl_plcl_pscs_space)))))

        # model 4
        self.cscl_plcl_pscs_lt_space = self.__construct_lt() | self.cscl_plcl_pscs_space        # Label Trigrams Space
        self.coo_4 = dict(zip(self.cscl_plcl_pscs_lt_space.keys(), list(range(len(self.cscl_plcl_pscs_lt_space)))))

    def train(self, ):
        w_1 = np.zeros(len(self.cscl_space), dtype=np.int64)
        w_2 = np.zeros(len(self.plcl_space), dtype=np.int64)
        w_3 = np.zeros(len(self.pscs_space), dtype=np.int64)
        w_4 = np.zeros(len(self.lt_space), dtype=np.int64)

        # plcl 5
        phi_2_sparse_dict = dict(zip([2, 3, 4, 5], [self.__phi_plcl_sparse(i) for i in [2, 3, 4, 5]]))
        # lt 5
        phi_4_sparse_dict = dict(zip([3, 4, 5], [self.__phi_lt_sparse(i) for i in [3, 4, 5]]))

        for piece in self.corpus:
            [x, y] = list(zip(*self.corpus[0]))
            candidate_y = self.label_space[len(x)]
            length = len(x)
            y_ind = candidate_y.index(y)

            # cscl
            phi_1_sparse, y_1_ind = self.__phi_cscl_sparse(x, y)
            score_1 = phi_1_sparse.dot(w_1)

            if length > 1:
                # plcl
                phi_2_sparse = phi_2_sparse_dict[length]
                score_2 = phi_2_sparse.dot(w_2)
                # pscs
                phi_3_sparse = self.__phi_pscs_sparse(x)
                score_3 = phi_3_sparse.dot(w_3)
            else:
                score_2 = 0
                score_3 = 0

            if length > 2:
                # lt
                phi_4_sparse = phi_4_sparse_dict[length]
                score_4 = phi_4_sparse.dot(w_4)
            else:
                score_4 = 0



    @staticmethod
    def __construct_candidate_label():
        label_dim_dict = {}
        max_dim = len(max(indexed_train_data, key=len))
        labels = [i for i in range(5)]
        for i in range(max_dim):
            label_dim_dict[i+1] = list(product(labels, repeat=(i + 1)))
        return label_dim_dict

    def __construct_cscl(self):
        cscl = Counter(list(chain.from_iterable(self.corpus)))
        # for key, count in dropwhile(lambda key_count: key_count[1] > 2, cscl.most_common()):
        #     del cscl[key]
        return cscl

    def __phi_cscl_sparse(self, x, y):
        candidate_y = self.label_space[len(x)]
        row = []
        col = []
        data = []
        for i in range(len(candidate_y)):
            cscl = set(zip(x, candidate_y[i]))
            phi_coo = self.cscl_space.keys() & cscl
            for coo in phi_coo:
                row.append(i)
                col.append(self.coo_1[coo])
                data.append(self.cscl_space[coo])

        y_sparse = csr_matrix((data, (row, col)), shape=(len(candidate_y), len(self.cscl_space)))
        return y_sparse, candidate_y.index(y)

    """ only call the max length of x times """
    def __construct_plcl(self):
        plcl = []
        labels = [list(zip(*i))[1] for i in self.corpus if len(i) > 1]
        for label in labels:
            if len(label) == 1:
                plcl.append(label)
            else:
                plcl.extend(list(zip(label, label[1:])))
        return Counter(plcl)

    def __phi_plcl_sparse(self, y_len):
        candidate_y = self.label_space[y_len]
        row = []
        col = []
        data = []
        for i in range(len(candidate_y)):
            y = candidate_y[i]
            plcl = set(zip(y, y[1:]))
            phi_coo = self.plcl_space.keys() & plcl
            for coo in phi_coo:
                row.append(i)
                col.append(self.coo_2[coo])
                data.append(self.plcl_space[coo])
        phi_2_sparse = csr_matrix((data, (row, col)), shape=(len(candidate_y), len(self.plcl_space)))
        return phi_2_sparse

    def __construct_pscs(self):
        pscs = []
        samples = [list(zip(*i))[0] for i in self.corpus if len(i) > 1]
        for sample in samples:
            if len(sample) == 1:
                pscs.append(sample)
            else:
                pscs.extend(list(zip(sample, sample[1:])))
        pscs = Counter(pscs)
        for key, count in dropwhile(lambda key_count: key_count[1] > 2, pscs.most_common()):
            del pscs[key]
        return pscs

    def __phi_pscs_sparse(self, x):
        candidate_y = self.label_space[len(x)]
        pscs = set(zip(x, x[1:]))
        phi_coo = pscs & self.pscs_space.keys()
        row = []
        col = []
        data = []
        for i in range(len(x)):
            for coo in phi_coo:
                row.append(i)
                col.append(self.coo_3[coo])
                data.append(self.pscs_space[coo])
        phi_3_sparse = csr_matrix((data, (row, col)), shape=(len(candidate_y), len(self.plcl_space)))
        return phi_3_sparse

    def __construct_lt(self):
        plcl = []
        labels = [list(zip(*i))[1] for i in self.corpus if len(i) > 2]
        for label in labels:
            if len(label) == 1:
                plcl.append(label)
            else:
                plcl.extend(list(zip(label, label[1:], label[2:])))
        return Counter(plcl)

    def __phi_lt_sparse(self, y_len):
        candidate_y = self.label_space[y_len]
        row = []
        col = []
        data = []
        for i in range(len(candidate_y)):
            y = candidate_y[i]
            plcl = set(zip(y, y[1:], y[2:]))
            phi_coo = self.lt_space.keys() & plcl
            for coo in phi_coo:
                row.append(i)
                col.append(self.coo_2[coo])
                data.append(self.lt_space[coo])
        phi_4_sparse = csr_matrix((data, (row, col)), shape=(len(candidate_y), len(self.lt_space)))
        return phi_4_sparse


# ################################################ #
#                   RUN FROM HERE                  #
# ################################################ #
if __name__ == '__main__':
    # ----------------- load data -----------------
    abs_path = os.path.abspath('.')
    train_data = __load_dataset_sents(os.path.join(abs_path, 'train.txt'))

    # ----------------- Preprocess Data -----------------
    # index label
    label_index = {'O': 0, 'PER': 1, 'LOC': 2, 'ORG': 3, 'MISC': 4}
    # index word and reconstruct train_data
    word_index, indexed_train_data = __preprocess_data(label_index, train_data)

    # ----------------- Train Model -----------------
    a = StructuredProceptron(indexed_train_data)
    # ----------------- Test Model -----------------


