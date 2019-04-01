import operator
from collections import Counter
import sys
import itertools
import numpy as np
import time
import random
from sklearn.metrics import f1_score
import collections
from itertools import chain


def load_dataset_sents(file_path, as_zip=True, to_idx=False, token_vocab=None, target_vocab=None):
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


# feature space of cw_ct
def cw_ct_counts(data, freq_thresh=5):
    # data inputted as (cur_word, cur_tag)
    cw_c1_c = Counter()
    for doc in data:
        cw_c1_c.update(Counter(doc))
    return Counter({k: v for k, v in cw_c1_c.items() if v > freq_thresh})


# feature representation of a sentence cw-ct
def phi_1(sent, cw_ct_counts):
    # sent as (cur_word, cur_tag)
    phi_1 = Counter()
    # include features only if found in feature space
    phi_1.update([item for item in sent if item in cw_ct_counts.keys()])
    return phi_1


""" ----------------- Perceprton -----------------"""


class Perceptron:
    def __init__(self, tags, feature_space, mod=1):
        super(Perceptron, self).__init__()
        self.feature_space = feature_space
        self.all_tags = tags
        if mod == 1:
            self.compute_score = self.scoring
        elif mod == 2:
            self.compute_score = self.viterbi
        else:
            self.compute_score = self.beam_search_scoring

    # creating all possible combinations of
    def pos_combos(self, sentence):
        combos = [list(zip(sentence, p)) for p in itertools.product(self.all_tags, repeat=len(sentence))]
        return combos

    def scoring(self, doc, weights, size=0):
        # unzip them
        sentence, tags = list(zip(*doc))
        # all possible combos of sequences
        combos = list(enumerate(self.pos_combos(sentence)))
        # our score matrix
        scores = np.zeros(len(combos))
        # looping through all possible combos
        for index, sent_tag in combos:
            # retrieving the counter if its in our feature space
            phi = phi_1(sent_tag, self.feature_space)
            # if its not then the score is 0
            if len(phi) == 0:
                scores[index] = 0
            else:
                temp_score = 0
                # otherwise do the w*local_phi
                for pair in phi:
                    if pair in weights:
                        temp_score += weights[pair] * phi[pair]
                    else:
                        temp_score += 0
                # store the score with the index
                scores[index] = temp_score
        # retrieve the index of the highest scoring sequence
        max_scoring_position = np.argmax(scores)
        # retrieve the highest scoring sequence
        max_scoring_seq = combos[max_scoring_position][1]
        return max_scoring_seq

    def viterbi(self, doc, weights, size=0):
        # unzip them
        sentence, _ = list(zip(*doc))
        # initialization: matrix V, B, max_scoring_seq
        matirix_v = np.zeros((len(self.all_tags), len(sentence)))
        matirix_b = np.zeros((len(self.all_tags), len(sentence)))
        max_scoring_seq = []
        # iteration
        for n in range(len(sentence)):
            for y in range(len(self.all_tags)):
                max_ = 0 if n == 0 else max(matirix_v[:, n - 1])
                w = 0 if (sentence[n], all_tags[y]) not in weights else weights[(sentence[n], self.all_tags[y])]
                argmax = 0 if n == 0 else np.argmax(matirix_v[:, n - 1])
                matirix_v[y, n] = max_ + w
                matirix_b[y, n] = argmax + w
        for n in range(len(sentence)):
            max_scoring_seq.append((sentence[n], self.all_tags[np.argmax(matirix_b[:, n])]))
        return max_scoring_seq

    def beam_search_scoring(self, doc, weight, size=3):
        # unzip them
        sentence, _ = list(zip(*doc))

        # start of sentence
        word = sentence[0]
        feat = list(zip([word] * len(self.all_tags), self.all_tags))
        w = np.array([weight[f] if f in weight else 0 for f in feat])
        beam = collections.OrderedDict(iter([(self.all_tags[l], w[l]) for l in w.argsort()[-size:]]))
        # sentence body
        for i in range(len(sentence))[1:]:
            beam_temp = collections.OrderedDict()
            for b in beam:
                feat = list(zip([sentence[i]] * len(self.all_tags), self.all_tags))
                w = np.array([weight[f] if f in weight else 0 for f in feat]) + beam[b]
                ll = [self.tuple_append(b, label) for label in self.all_tags]
                beam_temp.update(collections.OrderedDict(zip(ll, w)))
            # top 3
            iterms = list(beam_temp.items())
            np.random.shuffle(iterms)
            [ind, score] = zip(*iterms)
            # score = list(beam_temp.values())
            # ind = list(beam_temp.keys())
            beam = collections.OrderedDict(iter([(ind[l], score[l]) for l in np.argsort(score)[-size:]]))

        # return predicted label sequence
        labels = max(beam.items(), key=operator.itemgetter(1))[0]

        return list(zip(sentence, labels))

    @staticmethod
    def tuple_append(beam_key, label):
        if isinstance(beam_key, str):
            return beam_key, label
        if isinstance(beam_key, tuple):
            return (*beam_key, label)

    def train_perceptron(self, data, epochs, shuffle=True, size=0):
        # variables used as metrics for performance and accuracy
        iterations = range(len(data) * epochs)

        false_prediction = 0
        false_predictions = []
        # initialising our weights dictionary as a counter
        # counter.update allows addition of relevant values for keys
        # a normal dictionary replaces the key-value pair
        weights = Counter()
        # multiple passes
        for epoch in range(epochs):
            false = 0
            now = time.time()
            # going through each sentence-tag_seq pair in training_data
            # shuffling if necessary
            if shuffle:
                random.shuffle(data)
            for doc in data:
                # retrieve the highest scoring sequence
                max_scoring_seq = self.compute_score(doc, weights, size)
                # if the prediction is wrong
                if max_scoring_seq != doc:
                    correct = Counter(doc)
                    # negate the sign of predicted wrong
                    predicted = Counter({k: -v for k, v in Counter(max_scoring_seq).items()})
                    # add correct
                    weights.update(correct)
                    # negate false
                    weights.update(predicted)

                    """Recording false predictions"""
                    false += 1
                    false_prediction += 1
            false_predictions.append(false_prediction)
            print("Epoch: ", epoch + 1, " / Time for epoch: ", round(time.time() - now, 2),
              " / No. of false predictions: ", false)

            print("Evaluating the perceptron with (cur_word, cur_tag) \n")
            correct_tags, predicted_tags = perceptron.test_perceptron(test_data, weights)
            perceptron.evaluate(correct_tags, predicted_tags)
        return weights, false_predictions, iterations

    # testing the learned weights
    def test_perceptron(self, data, weights, size=0):
        correct_tags = []
        predicted_tags = []
        for doc in data:
            _, tags = list(zip(*doc))
            correct_tags.extend(tags)
            max_scoring_seq = self.compute_score(doc, weights, size)

            _, pred_tags = list(zip(*max_scoring_seq))
            predicted_tags.extend(pred_tags)
        return correct_tags, predicted_tags

    def evaluate(self, correct_tags, predicted_tags):
        f1_tags = ["PER", "LOC", "ORG", "MISC"]
        f1 = f1_score(correct_tags, predicted_tags, average='micro', labels=f1_tags)
        print("F1 Score: ", round(f1, 5))
        return f1


if __name__ == "__main__":
    random.seed(11242)
    depochs = 8
    feat_red = 0
    print("Default no. of epochs: ", depochs)
    print("Default feature reduction threshold: ", feat_red)
    print("Loading the data \n")

    """ ----------------- Loading the data -----------------"""
    train_data = load_dataset_sents("train.txt")
    test_data = load_dataset_sents("test.txt")

    # unique tags
    all_tags = ["O", "PER", "LOC", "ORG", "MISC"]

    """ ----------------- Defining our feature space ----------------- """
    print("Defining the feature space \n")
    cw_ct_count = cw_ct_counts(train_data, freq_thresh=feat_red)

    perceptron = Perceptron(all_tags, cw_ct_count, mod=1)
    print("Training the perceptron with (cur_word, cur_tag) \n")
    weights, false_predictions, iterations = perceptron.train_perceptron(train_data, epochs=depochs, size=5000)

    # print("Evaluating the perceptron with (cur_word, cur_tag) \n")
    # correct_tags, predicted_tags = perceptron.test_perceptron(test_data, weights)
    # perceptron.evaluate(correct_tags, predicted_tags)
