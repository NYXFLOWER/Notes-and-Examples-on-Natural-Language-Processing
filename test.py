import numpy as np
import timeit
import random

a = [i for i in range(10000)]
w = np.random.random(10000)

b = {}
for i in a:
    b[i] = i

list = [random.randint(0, 10000) for i in range(88)]

def npnp(w, list):
    return np.dot(w[list], list)

def didi(w, list):
    k = [w[i] for i in list]
    return np.dot(k, list)

def kkk(w, list):
    [np.dot(w[list], list) for i in range(400)]

def ppp(w, list):
    kkkk = np.zeros(shape=(10000, 400))
    for x in range(400):
        for y in range(88):
            kkkk[list[y], x] = list[y]
    np.dot(w, kkkk)