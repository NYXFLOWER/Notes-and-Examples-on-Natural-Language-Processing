import math
import os
import random
import re
import sys

# Complete the hackerlandRadioTransmitters function below.
# def hackerlandRadioTransmitters(x, k):


if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    nk = input().split()

    n = int(nk[0])

    k = int(nk[1])

    x = list(map(int, input().rstrip().split()))
    print(x)
    print(k)

    # result = hackerlandRadioTransmitters(x, k)

    # fptr.write(str(result) + '\n')

    # fptr.close()