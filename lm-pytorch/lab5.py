# -*- coding: utf-8 -*-
# Author: Robert Guthrie

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import OrderedDict


torch.manual_seed(1)
######################################################################

# word_to_ix = {"hello": 0, "world": 1}
# embeds = nn.Embedding(2, 5)  # 2 words in vocab, 5 dimensional embeddings
# lookup_tensor = torch.LongTensor([word_to_ix["hello"]])
# hello_embed = embeds(autograd.Variable(lookup_tensor))
# print(hello_embed)
#
# plt.plot(hello_embed[0, 0])
# plt.savefig("hhh.pdf")

######################################################################
# An Example: N-Gram Language Modeling
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Recall that in an n-gram language model, given a sequence of words
# :math:`w`, we want to compute
#
# .. math::  P(w_i | w_{i-1}, w_{i-2}, \dots, w_{i-n+1} )
#
# Where :math:`w_i` is the ith word of the sequence.
#
# In this example, we will compute the loss function on some training
# examples and update the parameters with backpropagation.
#
print("^^^^^^^^^^^^^^^^^^^^^^^^^ Setting ^^^^^^^^^^^^^^^^^^^^^^^^^^")
CONTEXT_SIZE = 2
EMBEDDING_DIM = 10
print("{:15s}: {:5d}".format("Context size", CONTEXT_SIZE))
print("{:15s}: {:5d}".format("Embedding dim", EMBEDDING_DIM))

# ----------- process data ----------
toy_dataset = ["The mathematician ran",
               "The mathematician ran to the store",
               "The physicist ran to the store",
               "The philosopher thought about it",
               "The mathematician solved the open problem"]

vocab = set()
trigrams = []
for sen in toy_dataset:
    tokens = ["<s>"] + sen.split(" ") + ["</s>"]
    vocab.update(tokens)
    trigrams += [([tokens[i], tokens[i+1]], tokens[i+2]) for i in range(len(tokens)-2)]
vocab = list(vocab)
vocab.sort()

word_to_ix = OrderedDict({word: i for i, word in enumerate(vocab)})
ix_to_word = list(word_to_ix.keys())


class NGramLanguageModeler(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramLanguageModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs

def compute_log_probs(con):
    # Step 1. Prepare the inputs to be passed to the model (i.e, turn the words
    # into integer indices and wrap them in variables)
    con_idxs = [word_to_ix[w] for w in con]
    con_var = autograd.Variable(torch.LongTensor(con_idxs))

    # Step 2. Recall that torch *accumulates* gradients. Before passing in a
    # new instance, you need to zero out the gradients from the old
    # instance
    model.zero_grad()

    # Step 3. Run the forward pass, getting log probabilities over next
    # words
    return model(con_var)


def pred_token(log_prob_tensor):
    """ :return (str) the word which has the highest probability"""
    probs = log_prob_tensor.detach().numpy()[0]
    ind = np.argmax(probs)

    assert not isinstance(ind, int)
    return ix_to_word[ind]


# hyper-parameter setting
learning_rate = 0.05
epoch_num = 61

print("{:15s}: {:5.2f}".format("Learning rate", learning_rate))
print("{:15s}: {:5d}".format("epoch number", epoch_num))
print()

loss_function = nn.NLLLoss()
model = NGramLanguageModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)

# ---------------------- train model -----------------------
losses = []
m = []
correct_san = []
boolean_test = []
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
np.random.seed(23432)
for epoch in range(epoch_num):
    total_loss = torch.Tensor([0])
    np.random.shuffle(trigrams)
    count_san = 0
    for context, target in trigrams:
        log_probs = compute_log_probs(context)

        # Step 4. Compute your loss function. (Again, Torch wants the target
        # word wrapped in a variable)
        loss = loss_function(log_probs, autograd.Variable(
            torch.LongTensor([word_to_ix[target]])))

        # Step 5. Do the backward pass and update the gradient
        loss.backward()
        optimizer.step()

        total_loss += loss.data
    losses.append(total_loss)
#
#
#
#     # ---------------------- Sanity check ---------------------
#     pred_binary = []
#     for x, y in trig_san:
#         probs_tensor = compute_log_probs(x)
#         y_pred = pred_token(probs_tensor)
#         if y_pred == y:
#             pred_binary.append(1)
#     correct_san.append(sum(pred_binary))
    # ---------------------- Sanity check ---------------------
    # probs = [0, 0]
    # for i in [0, 1]:
    #     for x, y in fill_sent(candi[i]):
    #         probs_tensor = compute_log_probs(x)
    #         probs[i] += probs_tensor[0, word_to_ix[y]].data
    # boolean_test.append(probs[0] > probs[1])
#
# print("loss", losses[0], losses[-1])  # The loss decreased every iteration over the training data!
#
# print("max correct number {:d} at {:d}".format(max(correct_san), np.argmax(correct_san)))

# ----------------------- test data -----------------------



# ---------------------------------------------------------
# ---------------------- Sanity check ---------------------
# ---------------------------------------------------------
print("^^^^^^^^^^^^^^^^^^^^^^^^^ Sanity Check ^^^^^^^^^^^^^^^^^^^^^^^^^^")
# data
test_san = ["<s>"] + "The mathematician ran to the store".split(" ") + ["</s>"]
trig_san = [([test_san[i], test_san[i + 1]], test_san[i + 2]) for i in range(len(test_san) - 2)]

line = "|---------------------------------------------------------------|"
form = "|{:15s}|{:15s}|{:15s}|{:15}|"
print(line)
print("|{:31s}|{:15s}|{:15}|".format("Input Word", "Predict Word", "Is Correct"))
print(line)

for x, y in trig_san:
    probs_tensor = compute_log_probs(x)
    y_pred = pred_token(probs_tensor)
    if y_pred == y:
        print(form.format(x[0], x[1], y_pred, "Yes"))
print(line)
print()

# # ---------------------------------------------------------
# # -------------------------- Test -------------------------
# # ---------------------------------------------------------
# data
def fill_sent(gap):
    return [(["<s>", "The"], gap), (["The", gap], "solved"), ([gap, "solved"], "the")]


candi = ["physicist", "philosopher"]
form = "-> {:13s} is: {:6.2f}"

print("^^^^^^^^^^^^^^^^^^^^^^^^^ Test ^^^^^^^^^^^^^^^^^^^^^^^^^^")

# comput log prob for physicist and philosopher
probs = [0, 0]
for i in [0, 1]:
    for x, y in fill_sent(candi[i]):
        probs_tensor = compute_log_probs(x)
        probs[i] += probs_tensor[0, word_to_ix[y]].data
print("The log prob of sentence with")
print(form.format('\'physicist\'', probs[0]))
print(form.format('\'philosopher\'', probs[1]))

choose = 0 if probs[0] > probs[1] else 1
print("Choose \'{:9s}\' fill the gap.".format(candi[choose]))
print()

lookup_tensor = [torch.LongTensor([word_to_ix[i]]) for i in candi+["mathematician"]]
embed = [model.embeddings(autograd.Variable(i)) for i in lookup_tensor]

print("The cosine similarity between 'mathematician' and")
cos = [F.cosine_similarity(embed[i], embed[2])[0] for i in [0, 1]]
print(form.format('\'physicist\'', cos[0]))
print(form.format('\'philosopher\'', cos[1]))
choose = 0 if cos[0] > cos[1] else 1
print("Choose \'{:9s}\' fill the gap.".format(candi[choose]))
