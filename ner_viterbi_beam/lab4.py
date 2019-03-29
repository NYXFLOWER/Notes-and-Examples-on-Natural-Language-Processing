import  argparse, time
import numpy as np
import matplotlib.pyplot as plt

from collections import Counter
from itertools import  product
from sklearn.metrics import f1_score
from scipy.sparse import coo_matrix

# this function provided by assignment requirement is to load data and format data
def load_dataset_sents(file_path, as_zip=True, to_idx=False, token_vocab=None, target_vocab=None):
    targets = []
    inputs = []
    zip_inps = []
    with open(file_path) as f:
        for line in f:
            sent, labels = line.split('\t')
            words = [token_vocab[w.strip()] if to_idx else w.strip() for w in sent.split()]
            ner_labels = [target_vocab[w.strip()] if to_idx else w.strip() for w in labels.split()]
            inputs.append(words)
            targets.append(ner_labels)
            zip_inps.append(list(zip(words, ner_labels)))
    return zip_inps if as_zip else (inputs, targets)


def phi_1(x, y):
    return dict(Counter([d[0]+ "_" + d[1] for d in zip(x,y)]))


# to get the unique words dictionary which the value is index of word in matrix
# and unique labels dictionary which the value is index of labels in matrix
def get_keys_for_phi1(data):
    words = []
    labels= []
    keys = []
    for s in data:
        for term in s:
            words.append(term[0])
            labels.append(term[1])
    words = np.unique(words)
    labels = np.unique(labels)

    i = 0
    for w in words:
        keys.extend((w+"_"+l, i) for (l,i) in zip(labels,range(i, i+len(labels))))
        i+=len(labels)

    return dict(keys)


# training function
def train(train_data,  weight, keys, labels, iterate_num, phi , prediction_fn, valid_func = None):
    weight_sum  = weight
    f1_scores = []
    for i in range(iterate_num):
        start = time.clock()
        np.random.seed(i)
        train_data = np.random.permutation(train_data)
        for j in range(len(train_data)):
            prediction = prediction_fn(weight, train_data[j][0], labels, keys)          # prediction = [labels,....]
            update = not prediction == train_data[j][1]
            if update:
                weight = update_weight(weight, train_data[j], prediction, keys, phi)
            # break
        weight_sum += weight  # sum

        print('Cost of %d th iteration: %.2fs ' % ( i+1 ,(time.clock() - start)))
        if(valid_func):
            f1_score = valid_func(weight, labels, keys)
            print("f1_score: %f \n" % f1_score)
            f1_scores.append(f1_score)

        start = time.clock()

    return weight_sum / iterate_num, f1_scores


def update_weight(weight, corrected, predicted, keys, phi):
    co_dict = phi(corrected[0],corrected[1])
    pr_dict = phi(corrected[0],predicted)

    for k in co_dict:
        weight[keys[k]] += co_dict[k]
    for k in pr_dict:
        weight[keys[k]] -= pr_dict[k]
    #
    return weight


# get the labels in the data set
def get_labels(data):
    labels = []
    for s in data:
        for term in s:
            labels.append(term[1])

    labels = np.unique(labels)
    return labels


# viterbi predict function
def viterbi_predict(weight, sentence,  labels, keys):

    current_pro = {}
    path = dict((l, []) for l in labels)
    for l in labels:
        current_pro[l] =  weight[keys[sentence[0] + "_" + l]] if sentence[0] + "_" + l in keys else 0

    for i in range(1, len(sentence)):
        last_pro = current_pro
        current_pro = {}
        for l in labels:
            pro, mxlabel =max( [( last_pro[la] + (weight[keys[sentence[i] + "_" + l]] if sentence[i] + "_" + l in keys else 0) , la ) for la in labels] , key= lambda x:x[0])
            current_pro[l] = pro
            path[l].append(mxlabel)

    max_pro = -1
    max_path = None
    for label in current_pro:
        path[label].append(label)
        if current_pro[label] > max_pro:
            max_path = path[label]
            max_pro = current_pro[label]


    return max_path


# format the data for training, ([sentence],[labels])
def get_format_data(train_data):
    data = []
    for s in train_data:
        x = []
        y = []
        for term in s:
            x.append(term[0])   # sentence
            y.append(term[1])   # labels
        data.append((tuple(x),tuple(y)))
    return tuple(data)


# beam search prediction function
def beam_prediction(size = 1):
    def prediction_fn(weight, sentence, labels, keys):

        B = [{"labels": (), "score": 0} for i in range(size)]
        for w in sentence:
            B_ = []
            for b in B:
                for l in labels:
                    if w + "_" + l in keys:
                        B_.append(
                            {"labels": b["labels"] + (l,), "score": (weight[keys[w + "_" + l]] + b["score"])})  # phi1
                    else:
                        B_.append({"labels": b["labels"] + (l,), "score": (-1 + b["score"])})  # unseen pairs given -1

            if len(B_) <= size:
                B = B_
            else:
                B_.sort(key=lambda e: e["score"], reverse=True)
                B = B_[:size]
        return B[0]["labels"]


    return prediction_fn


def valid_data(test_data, prediction_fn):

    correct = []
    predicted = []

    def valid_ (weight, labels, keys):
        for j in range(len(test_data)):
            prediction = prediction_fn(weight, test_data[j][0], labels, keys)
            correct.extend([_ for _ in test_data[j][1]])
            predicted.extend([_ for _ in prediction])
        return f1_score(correct, predicted, average='micro', labels=['ORG', 'MISC', 'PER', 'LOC'])

    return  valid_


#------------------- main ---------------------
parser = argparse.ArgumentParser()
parser.add_argument('-v',  required=False, action="store_true")
parser.add_argument('-b',  required=False, action="store_true")
parser.add_argument('train_file', type=str, help="Training  file")
parser.add_argument('test_file', type=str, help="Testing file")
# parser.add_argument('-t', type=str, help="term", default="all", required=False)
args = parser.parse_args()

train_file = args.train_file
test_file = args.test_file
run_viterbi = args.v
run_beam = args.b

# load data
train_data = load_dataset_sents(train_file)
test_data = load_dataset_sents(test_file)
labels = get_labels(train_data)

# format data
format_data = get_format_data(train_data)
test_data = get_format_data(test_data)
phi1_keys = get_keys_for_phi1(train_data)
weight = np.zeros((len(phi1_keys.keys()), 1)) # to give it a initial value

epochs = 10

### for viterbi
if run_viterbi:
    weight, f1 = train(format_data, weight, phi1_keys, labels, epochs, phi_1, viterbi_predict,
                       valid_func=valid_data(test_data, viterbi_predict))

### for Beam search
if run_beam:
    beam_size = 1
    beam_search = beam_prediction(beam_size)
    weight, f1 = train(format_data, weight, phi1_keys, labels, epochs, phi_1, beam_search,
                       valid_func=valid_data(test_data, beam_search))
