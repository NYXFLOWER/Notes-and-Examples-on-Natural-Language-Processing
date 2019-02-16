"""
@InProceedings{Pang+Lee:04a,
  author =       {Bo Pang and Lillian Lee},
  title =        {A Sentimental Education: Sentiment Analysis Using Subjectivity Summarization Based on Minimum Cuts},
  booktitle =    "Proceedings of the ACL",
  year =         2004
}
"""
import math
import multiprocessing
# from argparse import ArgumentParser
from collections import Counter
# from tqdm import tqdm
from enum import Enum
from itertools import dropwhile
import re
# import argparse
import numpy as np
# import matplotlib.pyplot as plt
import spacy
import os
import time
from line_profiler import LineProfiler

lp = LineProfiler()
prf_dic = {}


class Model(Enum):
    Binary = "Binary"
    Unigram = "Unigram"


def prf(func):
    if func.__name__ not in prf_dic:
        prf_dic[func.__name__] = lp(func)
    return prf_dic[func.__name__]


def show_profile_log():
    lp.print_stats()


def list_path(directory_path):
    neg_file_list, pos_file_list = os.listdir("%sneg/" % directory_path), os.listdir("%spos/" % directory_path)
    train_sample = pos_file_list[:800]
    train_sample.extend(neg_file_list[:800])
    test_sample = pos_file_list[800:]
    test_sample.extend(neg_file_list[800:])

    return train_sample, test_sample


if __name__ == '__main__':
    path = './review_polarity' + '/txt_sentoken/'
    

