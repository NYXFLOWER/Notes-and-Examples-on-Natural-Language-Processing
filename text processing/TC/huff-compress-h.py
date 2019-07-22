###############################################################################
# coding: utf-8
# Author: Hao Xu
# ID: 180127472
###############################################################################
import argparse, re, array, bisect, pickle, os.path
from collections import Counter


class Node:
    def __init__(self, symbol=None, count=0, left=None, right=None):
        self.symbol = symbol
        self.count = count
        self.left = left
        self.right = right


class HuffmanCompress:
    def __init__(self, symbol_model, input_file):
        self.file = open(input_file, encoding="utf-8").read()
        self.symbol_model = symbol_model
        self.tree_head = None
        self.code_tree()

    def __init_counts__(self):
        if self.symbol_model == 'word':
            self.words = Counter(re.split(r'(\W)', self.file))
            self.words[''] = 1
            self.words = self.words.most_common()
            # self.add_symbol_count_to_dict()
        else:
            self.words = Counter(self.file)
            self.words[''] = 1
            self.words = self.words.most_common()

    def code_tree(self):
        self.__init_counts__()
        # sorted_words = self.words
        n_word = len(self.words)
        # create node list
        nodes = [Node(self.words[j][0], self.words[j][1]) for j in range(n_word)][::-1]
        keys_bisect = [k.count for k in nodes]
        # build tree
        while len(nodes) > 1:
            temp1, temp2 = nodes[0], nodes[1]
            new_count = temp1.count + temp2.count
            new_index = bisect.bisect_left(keys_bisect, new_count)
            nodes.insert(new_index, Node(count=new_count, left=temp1, right=temp2))
            nodes.remove(temp1), nodes.remove(temp2)  # remove 2
        self.tree_head = nodes[0]

    @staticmethod
    def compress(string_list, code_array, huffman_code, type):
        n_string = len(string_list)
        temp = ''
        if type == 'word':
            for i in range(n_string):
                if string_list[i] == '':
                    continue
                code = huffman_code.get(string_list[i])
                for char in code:
                    if len(temp) == 8:
                        code_array.append(int(temp, 2))
                        temp = ''
                    temp += char
        else:
            for char in string:
                if char == '':
                    continue
                code = huffman_code.get(char)
                for num in code:
                    if len(temp) == 8:
                        code_array.append(int(temp, 2))
                        temp = ''
                    temp += num
        if len(temp) < 8:
            temp += '0' * (8 - len(temp))
        code_array.append(int(temp, 2))

    @staticmethod
    def get_huff_code(node, dict, prefix):
        if node.symbol is not None:
            dict[node.symbol] = prefix  # bin(int(prefix, 2))
            return
        HuffmanCompress.get_huff_code(node.left, dict, prefix + str(0))
        HuffmanCompress.get_huff_code(node.right, dict, prefix + str(1))


###############################################################################
# Process command line args
###############################################################################
parser = argparse.ArgumentParser(description='Huffman Coding Text Compression: Main Function')
parser.add_argument("infile", help="pass infile to huff-compress for compression")
parser.add_argument("-s", "--symbolmodel",
                    help="specify character- or word-based Huffman encoding -- default is character",
                    choices=["char", "word"])
args = parser.parse_args()

if not args.symbolmodel:
    symbolmodel = "char"
else:
    symbolmodel = args.symbolmodel

###############################################################################
# read from input file
###############################################################################
with open(args.infile, encoding="utf-8") as my_file:
    if symbolmodel == 'word':
        string = re.split(r'(\W)', my_file.read())
    else:
        string = my_file.read()

###############################################################################
# build huffman tree
###############################################################################
huffman_code = {}
h = HuffmanCompress(symbolmodel, args.infile)
h.get_huff_code(h.tree_head, huffman_code, '')

###############################################################################
# compress file
###############################################################################
compressed_binary = array.array('B')
h.compress(string, compressed_binary, huffman_code, symbolmodel)

###############################################################################
# write tree head to pkl file and write compressed huffman code to binary file
###############################################################################
(root, file) = os.path.splitext(args.infile)
with open(root +'.bin','wb') as wf:
    compressed_binary.tofile(wf)
with open(root +"-symbol-model.pkl", 'wb') as pf:
    # pickle.dump(huffman_code, pf)
    pickle.dump(h.tree_head, pf)

