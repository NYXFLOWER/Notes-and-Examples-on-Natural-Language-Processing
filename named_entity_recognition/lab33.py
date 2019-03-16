"""
Created by Hao Xu -- 10th Mar
This code runs and tests on MacOS 10.13.6

Command for run the code:
    $ python3 lab3.py train.txt test.txt

The data used in this code is from the following link.

"""
import os
from time import time
from collections import Counter
from itertools import product, chain, dropwhile
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics import f1_score
from line_profiler import LineProfiler
import matplotlib.pyplot as plt


lp = LineProfiler()
prf_dic = {}


def prf(func):
    if func.__name__ not in prf_dic:
        prf_dic[func.__name__] = lp(func)
    return prf_dic[func.__name__]


def show_profile_log():
    lp.print_stats()


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


def __construct_candidate_label(max_dim=5):
    l = ['a', 'b', 'c', 'd', 'e']
    label_dim_dict = {}
    max_dim = len(max(indexed_train_data, key=len))
    for i in range(max_dim):
        label_dim_dict[i + 1] = list(product(l, repeat=(i + 1)))
    return label_dim_dict


def __construct_cscl(corpus):
    cscl = Counter(list(chain.from_iterable(corpus)))
    for key, count in dropwhile(lambda key_count: key_count[1] > 2, cscl.most_common()):
        del cscl[key]
    return cscl


def __construct_plcl(corpus):
    plcl = []
    labels = [list(zip(*i))[1] for i in corpus if len(i) > 1]
    for label in labels:
        if len(label) == 1:
            plcl.append(label)
        else:
            plcl.extend(list(zip(label, label[1:])))
    return Counter(plcl)


def __construct_pscs(corpus):
    pscs = []
    samples = [list(zip(*i))[0] for i in corpus if len(i) > 1]
    for sample in samples:
        if len(sample) == 1:
            pscs.append(sample)
        else:
            pscs.extend(list(zip(sample, sample[1:])))
    pscs = Counter(pscs)
    for key, count in dropwhile(lambda key_count: key_count[1] > 2, pscs.most_common()):
        del pscs[key]
    return pscs


def __train(corpus, feature_space, label_space_dict, mod=1, max_epoch=5):
    num_feature = len(feature_space)
    num_sample = len(corpus)
    w = dict(zip(feature_space.keys(), [0]*num_feature))
    sample_index_list = [i for i in range(num_sample)]

    for i in range(max_epoch):
        np.random.shuffle(sample_index_list)
        for sample_index in sample_index_list:
            [x, y_true] = list(zip(*(corpus[sample_index])))
            label_space = label_space_dict[len(x)]

            # true y
            feature_set_true = set(zip(x, y_true)) & feature_space.keys()
            score_true = sum([w[feature] * feature_space[feature] for feature in feature_set_true])

            # candidate y
            y_pred = y_true
            current_highest_score = 0
            y_pred_feature = set()
            for y_candidate in label_space:
                feature_set = set(zip(x, y_candidate)) & feature_space.keys()
                score = sum([w[feature] * feature_space[feature] for feature in feature_set])
                if score > current_highest_score:
                    y_pred = y_candidate
                    current_highest_score = score
                    y_pred_feature = feature_set
            # compare and update w
            if current_highest_score > score_true:
                for feature in feature_set_true:
                    w[feature] += feature_space[feature]
                for feature in y_pred_feature:
                    w[feature] -= feature_space[feature]
    return w


# ----------------------------------------------------------------------------------------------- #
def __train_structured_proceptron_with_sparse_matrix(corpus, feature_space, label_space_dict, mod=1, max_epoch=5):
    """ mod: 1, 2, 3 """
    num_feature_space = len(feature_space)
    num_corpus = len(corpus)
    featrue_keys = set(feature_space.keys())

    np.random.seed(10)
    w = csr_matrix((num_feature_space, 1), dtype=np.int64)
    w_sum = csr_matrix((num_feature_space, 1), dtype=np.float64)

    w_out = []

    for i in range(max_epoch):

        np.random.shuffle(corpus)
        for piece in corpus:
            [x, y] = list(zip(*piece))
            label_space = label_space_dict[len(x)]
            phi_sparse = __feature_sparse_all(x, y, feature_space, featrue_keys, label_space, mod)
            pred_ind = __predict_label(phi_sparse, w)
            # pred_ind, phi_sparse = __train_piece(x, y, label_space, feature_space, mod, w)
            y_ind = label_space.index(y)
            if y_ind != pred_ind:
                w += (phi_sparse.getrow(y_ind) - phi_sparse.getrow(pred_ind)).reshape((num_feature_space, 1))

            w_sum += w
        # w_out.append(w_sum / (i+1) / num_corpus)
        w_out.append(w)

    return w_out


def __test(corpus, w, feature_space, label_space_dict, mod=1):
    true_label = []
    pred_label = []
    feature_keys = set(feature_space.keys())

    for piece in corpus:
        [x, y] = list(zip(*piece))
        label_space = label_space_dict[len(x)]

        phi_sparse = __feature_sparse_all(x, y, feature_space, feature_keys, label_space, mod)
        pred_ind = __predict_label(phi_sparse, w)
        y_ind = label_space.index(y)
        true_label.extend(label_space[y_ind])
        pred_label.extend(label_space[pred_ind])

    score = f1_score(true_label, pred_label, average='micro', labels=[0, 1, 2, 3, 4])
    return score


def __feature_sparse_all(x, y, feature_dict, feature_keys, label_space, mod):
    coo_dict = dict(zip(feature_dict.keys(), list(range(len(feature_dict)))))
    row, col, data = [], [], []
    for i in range(len(label_space)):
        feature_set = set(zip(x, label_space[i]))  # cscl
        if mod == 2:
            feature_set = feature_set | set(zip(y, y[1:]))  # add plcl
        if mod == 3:
            feature_set = feature_set | set(zip(y, y[1:]))  # add plcl
            feature_set = feature_set | set(zip(x, x[1:]))  # add pscs
        phi_coo = feature_set & feature_keys
        for coo in phi_coo:
            row.append(i)
            col.append(coo_dict[coo])
            data.append(feature_dict[coo])

    feature_sparse = csr_matrix((data, (row, col)), shape=(len(label_space), len(feature_dict)))

    return feature_sparse


def __predict_label(phi_sparse, w):
    score = phi_sparse.dot(w)
    pred_ind = np.argmax(score)

    return pred_ind


# ################################################ #
#                   RUN FROM HERE                  #
# ################################################ #
if __name__ == '__main__':
    # ----------------- load data -----------------
    abs_path = os.path.abspath('.')
    train_data = __load_dataset_sents(os.path.join(abs_path, 'train.txt'))

    # ----------------- Preprocess Data -----------------
    # index label
    label_index = {'O': 'a', 'PER': 'b', 'LOC': 'c', 'ORG': 'd', 'MISC': 'e'}
    # index word and reconstruct train_data
    word_index, indexed_train_data = __preprocess_data(label_index, train_data)

    # construct indexed test data
    test_data = __load_dataset_sents(os.path.join(abs_path, 'test.txt'))
    indexed_test_data = []
    for sent in test_data:
        temp = []
        for (w, l) in sent:
            if w in word_index:
                w = word_index[w]
            temp.append((w, label_index[l]))
        indexed_test_data.append(temp)

    # ----------------- Construct Feature Space -----------------
    # Candidate Label Space
    candidate_label = __construct_candidate_label()

    # Current Sample-Current Label Feature Space
    fspace_1 = __construct_cscl(indexed_train_data)

    # Previous Label-Current Label & Current Sample-Current Label Feature Space
    fspace_2 = __construct_plcl(indexed_train_data) | fspace_1

    # Previous Label-Current Label & Current Sample-Current Label & Previous Sample-Current Sample Feature Space
    fspace_3 = __construct_pscs(indexed_train_data) | fspace_2

    # ----------------- Train Model -----------------
    start = time()
    # w_1 = __train(indexed_train_data, fspace_1, candidate_label, mod=1, max_epoch=1)

    t = time()
    w_1 = __train_structured_proceptron_with_sparse_matrix(indexed_train_data, fspace_1, candidate_label, mod=1, max_epoch=5)
    print("Time cost: ", (time() - t))
    t = time()
    w_2 = __train_structured_proceptron_with_sparse_matrix(indexed_train_data, fspace_2, candidate_label, mod=2, max_epoch=5)
    print("Time cost: ", (time() - t))
    t = time()
    w_3 = __train_structured_proceptron_with_sparse_matrix(indexed_train_data, fspace_3, candidate_label, mod=3, max_epoch=5)
    print("Time cost: ", (time() - t))

    # ----------------- Test Model ------------------
    score_1, score_2, score_3 = [], [], []
    for i in range(5):
        score_1.append(__test(indexed_test_data, w_1[i], fspace_1, candidate_label, mod=1))
        score_2.append(__test(indexed_test_data, w_2[i], fspace_2, candidate_label, mod=2))
        score_3.append(__test(indexed_test_data, w_3[i], fspace_3, candidate_label, mod=3))

    # print(score_1)
    print("Time cost: ", (time() - start))

    print("score 1: ", score_1)
    print("score 2: ", score_2)
    print("score 3: ", score_3)

    # ----------------- Plot Figure -----------------
    plt.figure()
    xxx = [i + 1 for i in range(5)]
    title = "f1 score over test dataset during 5 epochs."
    plt.plot(xxx, score_1, 'r', label='cscl')
    plt.plot(xxx, score_2, 'g', label='cscl & plcl')
    plt.plot(xxx, score_3, 'b', label='cscl & plcl & pscs')
    plt.xticks(xxx)
    plt.ylim((0, 1))
    plt.title(title)
    plt.grid()
    plt.legend()
    plt.savefig("lab33_hao_xu.pdf")
    plt.show()

# show_profile_log()
#     w = [0 for i in range(len(fspace_3))]
#     socre = __test(indexed_test_data, w, fspace_3, candidate_label, mod=1)
