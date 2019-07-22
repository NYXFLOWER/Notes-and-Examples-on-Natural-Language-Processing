#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 15:41:37 2018

test-harness.py: a script to test huffman compression and decompression programs
    for input and ouput command line and file naming conventions as specified
    in the COM4115/6115 Text Compression Assignment

Change History:
    v1.0 - initial release
    v1.1 - changed to run on Windows

@author: R.Gaizauskas
"""
import time

version_number = "1.1"
timeout = 12000

import argparse
import subprocess
from subprocess import run
import os.path
import sys

# Process command line args
parser = argparse.ArgumentParser()
parser.add_argument("-v", "--version", help="shows version number",
                    action="store_true")
parser.add_argument("infile", help="pass infile to huff-compress/decompress for compression/decompression")
parser.add_argument("-s", "--symbolmodel", help="specify character- or word-based Huffman encoding -- default is character",
                    choices=["char","word"])
args= parser.parse_args()
if args.version:
    print("version: ",version_number)
    exit(1)

if not(args.symbolmodel):
    symbolmodel= "char" 
else:
    symbolmodel = args.symbolmodel

# Check that input file is there
try:
    f = open(args.infile,'rb')
    
except FileNotFoundError:   
    print("test-harness: Error: file",args.infile,"not found")
    exit(1)
f.close()


# Try to run huff-compress.py
# Will fail if:
# 1. huff-compress.py not found in current working directory
# 2. huff-compress.py times out -- time limit set to timeout seconds
# 3. for other reasons, e.g. arguments supplied to huff-compress by test-harness 
#    are not correctly handled by huff-compress
print("test-harness: testing huff-compress.py on ",args.infile," with ",symbolmodel," encoding ...")
a = time.time()
try:
    cp = run([sys.executable,"./huff-compress.py","-s",symbolmodel,args.infile],check=True,timeout=timeout)
    print(time.time() - a)
except FileNotFoundError:   
    print("test-harness: Error: file ./huff-compress.py not found")
    exit(1)

except subprocess.TimeoutExpired:
    print("test-harness: Error: huff-compress time limit (",timeout,"secs ) expired")
    exit(1)   

except subprocess.SubprocessError:
    print("test-harness: Error: huff-compress failed to run")
    print("\t Check usage: huff-compress.py [-h] [-s {char,word}] infile")
    exit(1)
    
#print(cp)
  
print("test-harness: huff-compress.py completed") 

# Check for huff-compress outputs.
# Will fail if: 
# 1. no infileroot-symbol-model.pkl file in current working directory
# 2. no infileroot.bin in file in current working directory
print("test-harness: checking for huff-compress outputs ...") 

(root,file) = os.path.splitext(args.infile)
try:
    f = open(root +'-symbol-model.pkl', 'rb')
    
except FileNotFoundError:   
    print("test-harness: Error: file ",root +'-symbol-model.pkl'," not found")
    exit(1)
f.close()

try:
    f = open(root +'.bin','rb')
    
except FileNotFoundError:   
    print("test-harness: Error: file ",root +'.bin'," not found")
    exit(1)
f.close()

print("test-harness: huff-compress outputs found") 


# Try to run huff-decompress.py
# Will fail if:
# 1. huff-decompress.py not found in current working directory
# 2. huff-decompress.py times out -- time limit set to timeout seconds
# 3. for other reasons, e.g. arguments supplied to it by test-harness are
#    not correctly handled

print("test-harness: testing huff-decompress.py on",root +'.bin'," ...")
a = time.time()
try:
    cp = run([sys.executable,"./huff-decompress.py",root +'.bin'],check=True,timeout=timeout)
    print(time.time() - a)
except FileNotFoundError:   
    print("test-harness: Error: file ./huff-decompress.py not found")
    exit(1)

except subprocess.TimeoutExpired:
    print("test-harness: Error: huff-decompress time limit (",timeout,"secs ) expired")
    exit(1)   

except subprocess.SubprocessError:
    print("test-harness: Error: huff-decompress failed to run")
    print("\t Check usage: huff-decompress.py [-h] infile")
    exit(1)

print("test-harness: huff-decompress.py completed") 

# Check for huff-decompress outputs.
# Will fail if: 
# 1. no infileroot-decompressed.txt file in current working directory
print("test-harness: checking for huff-decompress outputs ...") 

try:
    f = open(root +'-decompressed.txt', 'rb')
    
except FileNotFoundError:   
    print("test-harness: Error: file ",root +'-decompressed.txt'," not found")
    exit(1)
f.close()

print("test-harness: huff-decompress outputs found") 

print("test-harness: all tests completed successfully")



