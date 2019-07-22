###############################################################################
# coding: utf-8
# Author: Hao Xu
# ID: 180127472
###############################################################################
import argparse, pickle, os.path


class Node:
    def __init__(self, symbol=None, left=None, right=None):
        self.symbol = symbol
        self.left = left
        self.right = right


class HuffmanDecodeTree:
    def __init__(self, tree_head):
        self.decode_tree = tree_head
        self.binary_string = ''
        self.my_string = ''
        self.pointer = 0

    def binary_code_to_string(self, bin_code):
        n_code = len(bin_code)
        temp = ['{:08b}'.format(bin_code[i]) for i in range(n_code)]
        self.binary_string = ''.join(temp)

    def search_tree(self, node):
        if self.binary_string[self.pointer] == '0':
            node = node.left
        else:
            node = node.right

        # check if end of file
        if self.pointer == len(self.binary_string)-1:
            return
        self.pointer += 1

        if node.symbol is not None:
            self.my_string += node.symbol
            return
        else:
            self.search_tree(node)

    def decode(self):
        self.pointer = 0
        while 1:
            current_node = self.decode_tree
            self.search_tree(current_node)
            if self.pointer == len(self.binary_string)-1:
                break


###############################################################################
# process command line args
###############################################################################
parser = argparse.ArgumentParser()
parser.add_argument("binary_file", help="pass infile to huff-decompress for decompression")
args = parser.parse_args()

###############################################################################
# read Huffman Tree from pkl file
###############################################################################
tree = Node()
(root, file) = os.path.splitext(args.binary_file)
if os.path.getsize(root +'-symbol-model.pkl') > 0:
    with open(root +'-symbol-model.pkl', "rb") as f:
        unpickler = pickle.Unpickler(f)
        tree = unpickler.load()     # tree head

###############################################################################
# decode bin file
###############################################################################
h = HuffmanDecodeTree(tree)
with open(args.binary_file, 'br') as code_file:
    h.binary_code_to_string(code_file.read())
h.decode()

###############################################################################
# write decoded Huffman codes to file
###############################################################################
with open(root + "-decompressed.txt", 'w') as o:
    o.write(h.my_string)
