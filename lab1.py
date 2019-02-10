"""
@InProceedings{Pang+Lee:04a,
  author =       {Bo Pang and Lillian Lee},
  title =        {A Sentimental Education: Sentiment Analysis Using Subjectivity Summarization Based on Minimum Cuts},
  booktitle =    "Proceedings of the ACL",
  year =         2004
}
"""
import math
from argparse import ArgumentParser
from collections import Counter
from tqdm import tqdm
import re
import argparse
import numpy as np
import spacy
import os


class PerceptronClassifier:
    def __init__(self, directory_path, mod=0, each_num_train=800):
        self.x_train_index, self.x_test_index = [[]], [[]]
        self.x_train_value, self.x_test_value = [[]], [[]]
        self.feature_dict = {'': 0}
        self.feature_index = 1
        self.mod = mod
        self.prepare_data(directory_path)

    def prepare_data(self, directory_path):
        pbar = tqdm(total=2000)
        nlp = spacy.load('en')
        file_list = os.listdir(directory_path + 'neg/')
        file_list.extend(os.listdir(directory_path + 'pos/'))

        count_file, feature_index = 0, 0

        for file in file_list:
            path = (directory_path + 'neg/' + file) if count_file < 1000 else (directory_path + 'pos/' + file)
            with open(path, 'r') as f:
                doc = nlp(re.sub("[^\w]", " ", f.read()))
            # build the input feature matrix
            value_list, index_list = self.__doc_to_list_and_update_feature_dict(doc)
            if count_file < 800 or (999 < count_file < 1800):
                self.x_train_value.append(value_list)
                self.x_train_index.append(index_list)
            else:
                self.x_test_value.append(value_list)
                self.x_test_index.append(index_list)
            count_file += 1
            pbar.update(1)
        pbar.close()

    def __doc_to_list_and_update_feature_dict(self, doc):
        value_list, index_list = [1], [0]
        if self.mod == 0:
            for token in doc:
                lemma = token.lemma_
                if lemma == ' ':
                    continue
                elif lemma not in self.feature_dict:
                    self.feature_dict[lemma] = self.feature_index
                    value_list.append(1)
                    index_list.append(self.feature_index)
                    self.feature_index += 1
                else:
                    index = self.feature_index[lemma]
                    value_list[index] += 1
                    index_list.append(index)

        return value_list, index_list


