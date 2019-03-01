"""
Created by Hao Xu -- 27th February
This code runs and tests on MacOS 10.13.6

Command for run the code:
    $ python3 lab2.py news-corpus-500k.txt questions.txt

The data used in this code is from the following link.

"""
from collections import Counter
import numpy as np
import re
import time

N_ROW, N_COLUMN = 2, 10
D_TYPE_INT = np.dtype('uint8')
D_TYPE_FLOAT = np.dtype('float64')

# ################################################ #
#                   RUN FROM HERE                  #
# ################################################ #
if __name__ == '__main__':
    # abs_path = os.path.abspath('.')
    # corpus_path = os.path.join(abs_path, sys.argv[1])
    # question_path = os.path.join(abs_path, sys.argv[2])

    corpus_path = '/Users/nyxfer/Documents/GitHub/nlp/sentence_completion/news-corpus-500k.txt'
    question_path = '/Users/nyxfer/Documents/GitHub/nlp/sentence_completion/questions.txt'

    # corpus = "Colorado's six-run through threw went know weather through whether whether fourth , however , combined the sort of bad luck and poor play that dogged the Diamondbacks .Sport wagons combine nearly all the velocity and road-grabbing craftiness of a proper sport sedan with the family-friendly utility of a SUV / crossover .By the way - the first amendment is words also - Words to live by , words that guide , explain , and give meaning and purpose .Nothing would boost consumer confidence and banker confidence more than a consensus that property prices have really bottomed out .Instead , offers on fresh food have risen.We don't provide detail on ongoing operations".lower()

    # read from corpus to a string (line by line)
    t = time.time()
    with open(corpus_path, 'r') as f:
        s = f.readlines()
    corpus_list = []
    for line in s:      # add <s> and <\s> at the begin and end of each line
        corpus_list.append("<s>")
        corpus_list.extend(re.sub("[^\w]", ' ', line.lower()).split())     # convert to lower case
        corpus_list.append('</s>')
    print(time.time() - t)


    # corpus_list = re.sub("[^\w]", " ", corpus).split()

    # counter for Unigram and Bigram
    counter_unigram = Counter(corpus_list)
    counter_bigram = Counter(zip(corpus_list, corpus_list[1:]))

    # Construct 2-d array
    dimension = len(counter_unigram)
    coordinate_dict, index = {}, 0
    frequency_2d = np.zeros(shape=(dimension, dimension), dtype=np.dtype('uint8'))
    for key, value in counter_unigram.items():
        # for coordinate
        coordinate_dict[key] = index
        # for value
        frequency_2d[index, index] = value          # all diagonal values are for unigram
        index += 1
    for (w1, w2), value in counter_bigram.items():
        # get coordinate
        (r, c) = (coordinate_dict[w1], coordinate_dict[w2])
        # write to 2d array
        frequency_2d[r, c] = value

    # read from question file to three matrix
    # t = time.time()
    with open(question_path, 'r') as q:
        question_list = q.readlines()
    # corpus_list = []
    # for line in s:  # add <s> and <\s> at the begin and end of each line
    #     corpus_list.append("<s>")
    #     corpus_list.extend(re.sub("[^\w]", '', line.lower()).split())  # convert to lower case
    #     corpus_list.append('</s>')
    # print(time.time() - t)

    # question_list = ["I don't know ____ to go out or not . : weather/whether",
    #                  "We went ____ the door to get inside . : through/threw"]
    sub_pattern = re.compile(r'____')          # when print sub it to answer

    # question_sp_list = [question.split() for question in question_list]

    # candidate_word_list = [re.split("\/", question_sp[-1]) for question_sp in question_sp_list] # list of candidate list
    index = 0

    c1_list = []
    c2_list = []

    join_c1 = np.zeros(shape=(N_ROW, N_COLUMN), dtype=D_TYPE_FLOAT)
    join_c2 = np.zeros(shape=(N_ROW, N_COLUMN), dtype=D_TYPE_FLOAT)
    marg_c1 = []
    marg_c2 = []
    # marg_pre = []

    for question in question_list:
        sp = question.split()
        blank_index = sp.index("____")
        [c1, c2] = re.split("\/", sp[-1])
        c1_list.append(c1)
        c2_list.append(c2)

        coo_c1, coo_c2 = coordinate_dict[c1], coordinate_dict[c2]

        # Unigram
        marg_c1.append(frequency_2d[coo_c1, coo_c1])
        marg_c2.append(frequency_2d[coo_c2, coo_c2])

        # Bigram or with Smoothing
        pre = coordinate_dict[sp[blank_index - 1]]  # coordinate of previous word
        ne = coordinate_dict[sp[blank_index + 1]]  # coordinate of next word
        join_c1[:, index] = [frequency_2d[pre, coo_c1], frequency_2d[coo_c1, ne]]
        join_c2[:, index] = [frequency_2d[pre, coo_c2], frequency_2d[coo_c2, ne]]

        # marg_pre.append(frequency_2d[pre, pre])

        index += 1

        # pre = coordinate_dict[sp[blank_index - 1]]      # coordinate of previous word
        # ne = coordinate_dict[sp[blank_index + 1]]       # coordinate of next word
        # fre_pre_c1, fre_pre_c2 = frequency_2d[pre, coo_c1], frequency_2d[pre, coo_c2]
        # fre_c1_next, fre_c2_next = frequency_2d[coo_c1, ne], frequency_2d[coo_c2, ne]
        # pre = frequency_2d[pre, pre]                    # frequency of previous word
        #
        # score_c1 =  math.log()

    # print()
    answer_uni = np.where(marg_c1 > marg_c2, c1_list, c2_list)


    # bigram
    score_c1 = np.sum(np.log(join_c1 + 0.0001), axis=0) - np.log(marg_c1)
    score_c2 = np.sum(np.log(join_c2 + 0.0001), axis=0) - np.log(marg_c2)
    answer_bi = np.where(score_c1 > score_c2, c1_list, c2_list)
    # print()

    # bigram with smoothing
    score_c1 = np.sum(np.log(join_c1 + 1), axis=0) - np.log(np.array(marg_c1) + dimension)
    score_c2 = np.sum(np.log(join_c2 + 1), axis=0) - np.log(np.array(marg_c2) + dimension)
    answers_sm = np.where(score_c1 > score_c2, c1_list, c2_list)

    # for i in range(N_COLUMN):
    #     print('Question ', i+1, " :")
    #     print(' ', question_list[i])
    #     print("  -> Unigram: ", answer_uni[i],
    #           "  -> Bigram: ", answer_bi[i],
    #           "  -> Bigram with Smoothing: ", answers_sm[i])
    #
    # print(time.time() - t)

    answer = ['whether', 'through', 'piece', 'court', 'allowed',
              'check', 'hear', 'cereal', 'chews', 'sell']
    score_compare = ['<', '', '', '', '',
                     '', '', '', '', '']

# def compute_score(score_c1_list, score_c2_list):
#     diff = score_c1_list - score_c2_list
#     score = 0.
#     for d in diff:
#
