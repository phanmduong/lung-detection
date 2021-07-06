#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 19 11:16:10 2021

@author: phanmduong
"""
import numpy as np

import matplotlib.pyplot as plt

if __name__ == '__main__':
    file_path = './data_preprocessed/augmented/positives/1.3.6.1.4.1.14519.5.2.1.6279.6001.117040183261056772902616195387_2_0.npy'
    img = np.load(file=f'{file_path}');

    plt.imshow(img[60,:,:], cmap=plt.cm.bone)
    plt.show()

    print(img.shape)