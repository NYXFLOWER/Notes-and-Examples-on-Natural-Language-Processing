from collections import Counter
import itertools
import numpy as np
import time
import random
from sklearn.metrics import f1_score
import argparse
import copy


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
def phi_1(sent, cw_ct_count):
    # sent as (cur_word, cur_tag)
    phi1 = Counter()
    # include features only if found in feature space
    phi1.update([item for item in sent if item in cw_ct_count.keys()])
    return phi1


class Perceptron:
    """Perceprton"""
    def __init__(self, tags, feature_space, modes):
        super(Perceptron, self).__init__()
        self.feature_space = feature_space
        self.all_tags = tags
        self.mode = modes

    # creating all possible combinations of
    def pos_combos(self, sentence):
        combos = [list(zip(sentence, p)) for p in itertools.product(self.all_tags, repeat=len(sentence))]
        return combos
    
    def scoring(self, doc, weight):
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
                    if pair in weight:
                        temp_score += weight[pair]*phi[pair]
                    else:
                        temp_score += 0
                # store the score with the index
                scores[index] = temp_score
        # retrieve the index of the highest scoring sequence
        max_scoring_position = np.argmax(scores)
        # retrieve the highest scoring sequence
        max_scoring_seq = combos[int(max_scoring_position)][1]
        return max_scoring_seq

    def viterbi(self, doc, weight):
        """
        predict by viterbi
        :param doc: a sentence with (word,label) pairs
        :param weight: weights Counter
        :return: max scoring sequence
        """
        # unzip them
        sentence, _ = list(zip(*doc))
        # initialization: matrix V, B, max_scoring_seq
        matrix_v = np.zeros((len(self.all_tags), len(sentence)))
        matrix_b = np.zeros((len(self.all_tags), len(sentence)))
        max_scoring_seq = []
        # iteration
        for n in range(len(sentence)):
            for y in range(len(self.all_tags)):
                max_ = 0 if n == 0 else max(matrix_v[:, n - 1])
                argmax = 0 if n == 0 else np.argmax(matrix_v[:, n - 1])
                w = 0 if (sentence[n], all_tags[y]) not in weight else weight[(sentence[n], self.all_tags[y])]
                matrix_v[y, n] = max_ + w
                matrix_b[y, n] = argmax + w
        # find max scoring sequence form B
        for n in range(len(sentence)):
            max_scoring_seq.append((sentence[n], self.all_tags[np.argmax(matrix_b[:, n])]))
        return max_scoring_seq

    def beam_search(self, doc, weight, k=3):
        """
        predict by beam search
        :param doc: a sentence with (word,label) pairs
        :param weight: weights Counter
        :param k: beam size
        :return: max scoring sequence
        """
        sentence, _ = list(zip(*doc))
        b = [([], 0)]
        for w in sentence:
            b_ = []
            for beam in b:
                for l in self.all_tags:
                    b_.append((beam[0] + [(w, l)], weight[(w, l)] + beam[1]))
            b = sorted(b_, key=lambda i: i[1], reverse=True)[:k]
        return max(b, key=lambda i: i[1])[0]

    def predict(self, doc, weight, modes="", k=1):
        """
        decide which method would be used
        :param doc: a sentence with (word,label) pairs
        :param weight: weights Counter
        :param modes: "v" for viterbi, "b" for beam search, others for the original scoring
        :param k: beam size
        :return: max scoring sequence
        """
        if modes == "v":
            return self.viterbi(doc, weight)
        elif modes == "b":
            return self.beam_search(doc, weight, k)
        else:
            return self.scoring(doc, weight)

    def train_perceptron(self, data, epochs, shuffle=True, k=1):
        # variables used as metrics for performance and accuracy
        iteration = range(len(data)*epochs)
        false_prediction = 0
        false_predictions = []
        # initialising our weights dictionary as a counter
        # counter.update allows addition of relevant values for keys
        # a normal dictionary replaces the key-value pair
        weight = Counter()
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
                # max_scoring_seq = self.scoring(doc, weights)
                max_scoring_seq = self.predict(doc, weight, self.mode, k)
                # if the prediction is wrong
                if max_scoring_seq != doc:
                    correct = Counter(doc)
                    # negate the sign of predicted wrong
                    predicted = Counter({k: -v for k, v in Counter(max_scoring_seq).items()})
                    # add correct
                    weight.update(correct)
                    # negate false
                    weight.update(predicted)

                    """Recording false predictions"""
                    false += 1
                    false_prediction += 1
                false_predictions.append(false_prediction)
            print("Epoch: ", epoch+1, " / Time for epoch: ", round(time.time() - now, 2),
                  " / No. of false predictions: ", false)
        return weight, false_predictions, iteration
    
    # testing the learned weights
    def test_perceptron(self, data, weight, k=1):
        correct_tag = []
        predicted_tag = []
        for doc in data:
            _, tags = list(zip(*doc))
            correct_tag.extend(tags)
            # max_scoring_seq = self.scoring(doc, weights)
            max_scoring_seq = self.predict(doc, weight, self.mode, k)
            _, pred_tags = list(zip(*max_scoring_seq))
            predicted_tag.extend(pred_tags)
        return correct_tag, predicted_tag

    @staticmethod
    def evaluate(correct_tag, predicted_tag):
        f1 = f1_score(correct_tag, predicted_tag, average='micro', labels=["PER", "LOC", "ORG", "MISC"])
        print("F1 Score: ", round(f1, 5))
        return f1


if __name__ == "__main__":
    depochs = 5
    feat_red = 0
    print("Default no. of epochs: ", depochs)
    print("Default feature reduction threshold: ", feat_red)
    print("Loading the data \n")
    # read command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--viterbi", help="viterbi", action="store_true")
    parser.add_argument("-b", "--beam", help="beam search", action="store_true")
    parser.add_argument("train")
    parser.add_argument("test")
    args = parser.parse_args()
    if args.viterbi:
        mode = "v"
    elif args.beam:
        mode = "b"
    else:
        mode = ""
    random.seed(11242)
    """Loading the data"""
    train_data = load_dataset_sents(args.train)
    test_data = load_dataset_sents(args.test)
    # unique tags
    all_tags = ["O", "PER", "LOC", "ORG", "MISC"]
    """ Defining our feature space """
    print("Defining the feature space \n")
    cw_ct = cw_ct_counts(train_data, freq_thresh=feat_red)
    if args.viterbi:
        mode = "v"
        perceptron = Perceptron(all_tags, cw_ct, mode)
        print("Training the perceptron with (cur_word, cur_tag) \n")
        weights, _, _ = perceptron.train_perceptron(train_data, epochs=depochs)
        print("Evaluating the perceptron with (cur_word, cur_tag) \n")
        correct_tags, predicted_tags = perceptron.test_perceptron(test_data, weights)
        perceptron.evaluate(correct_tags, predicted_tags)
    elif args.beam:
        mode = "b"
        for beam_size in [1, 3, 5]:
            random.seed(11242)
            perceptron = Perceptron(all_tags, cw_ct, mode)
            print("Beam size: ", beam_size, "\n")
            print("Training the perceptron with (cur_word, cur_tag) \n")
            weights, _, _ = perceptron.train_perceptron(copy.deepcopy(train_data), epochs=depochs, k=beam_size)
            print("Evaluating the perceptron with (cur_word, cur_tag) \n")
            correct_tags, predicted_tags = perceptron.test_perceptron(test_data, weights, k=beam_size)
            perceptron.evaluate(correct_tags, predicted_tags)
    else:
        mode = ""
        perceptron = Perceptron(all_tags, cw_ct, mode)
        print("Training the perceptron with (cur_word, cur_tag) \n")
        weights, _, _ = perceptron.train_perceptron(train_data, epochs=depochs)
        print("Evaluating the perceptron with (cur_word, cur_tag) \n")
        correct_tags, predicted_tags = perceptron.test_perceptron(test_data, weights)
        perceptron.evaluate(correct_tags, predicted_tags)
