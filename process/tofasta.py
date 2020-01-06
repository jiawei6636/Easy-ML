# -*- coding: utf-8 -*-
import h5py
import time
import itertools
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def onehot2sequence():
    alpha = ['A', 'C', 'G', 'T']
    filename = './data/train.mat'
    with h5py.File(filename, 'r') as file:
        one_hot_data = file['trainxdata'][:, :, :10000]  # shape = (1000, 4, 4400000)
        label = file['traindata'] # shape = (919, 4400000)
        one_hot_data = np.transpose(one_hot_data, (2, 0, 1))  # shape = (4400000, 1000, 4)
        label = np.transpose(label, (1, 0)) # shape = (4400000, 919)

    with open('./data/sequence.fasta', 'w+') as f:
        for num, sequence in tqdm(zip(range(10000), one_hot_data), total=10000, desc='Processing...', ascii=True):
            sequence = [np.argmax(i) for i in sequence]
            sequence = [alpha[i.item()] for i in sequence]
            sequence = ''.join(sequence)
            f.write('> ' + 'num=' + str(num) + '\n')
            f.write(sequence+'\n')

    print("Process over!")

if __name__ == '__main__':
    onehot2sequence()