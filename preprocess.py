#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 11 10:35:19 2021

@author: phanmduong
"""

#!/usr/bin/env python
"""
Builds image data base as test, train, validation datasets
Run script as python create_images.py $mode
where mode can be 'test', 'train', 'val'
"""
import sys
import glob
import os
from joblib import Parallel, delayed
import pickle
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import SimpleITK as sitk

raw_image_path ='./data/raw/*/'
candidates_file = './data/candidates.csv'

class CTScan(object):
    """
    A class that allows you to read .mhd header data, crop images and 
    generate and save cropped images

    Args:
    filename: .mhd filename
    coords: a numpy array
    """
    def __init__(self, filename = None, coords = None, path = None):
        """
        Args
        -----
        filename: .mhd filename
        coords: coordinates to crop around
        ds: data structure that contains CT header data like resolution etc
        path: path to directory with all the raw data
        """
        self.filename = filename
        self.coords = coords
        self.ds = None
        self.image = None
        self.path = path
        

    def reset_coords(self, coords):
        """
        updates to new coordinates
        """
        self.coords = coords

    def read_mhd_image(self):
        """
        Reads mhd data
        """
        path = glob.glob(self.path + self.filename + '.mhd')
        
        if len(path) <= 0:
            return False
        
        self.ds = sitk.ReadImage(path[0])
        self.image = sitk.GetArrayFromImage(self.ds)
        return True

    def get_voxel_coords(self):
        """
        Converts Cartesian to voxel coordinates
        """
        origin = self.ds.GetOrigin()
        resolution = self.ds.GetSpacing()
        print("origin"+str(origin))
        print("resolution"+str(resolution))
        voxel_coords = [np.absolute(self.coords[j]-origin[j])/resolution[j] \
            for j in range(len(self.coords))]
        return tuple(voxel_coords)
    
    def get_subimage(self, width):
        """
        Returns cropped image of requested dimensions
        """
        if self.read_mhd_image() == False:
            return
        
        x, y, z = self.get_voxel_coords()
        subImage = self.image[int(z), int(y-width/2):int(y+width/2),\
         int(x-width/2):int(x+width/2)]
        return subImage   
    
    def normalizePlanes(self, npzarray):
        """
        Copied from SITK tutorial converting Houndsunits to grayscale units
        """
        maxHU = 400.
        minHU = -1000.
        npzarray = (npzarray - minHU) / (maxHU - minHU)
        npzarray[npzarray>1] = 1.
        npzarray[npzarray<0] = 0.
        return npzarray
    
    def save_image(self, filename, width):
        """
        Saves cropped CT image
        """
        image = self.get_subimage(width)
        
        if image is None:
            return;
        
        image = self.normalizePlanes(image)
        Image.fromarray(image*255).convert('L').save(filename)

def create_data(idx, outDir, X_data,  width = 50):
    """
    Generates your test, train, validation images
    outDir = a string representing destination
    width (int) specify image size
    """
    scan = CTScan(np.asarray(X_data.loc[idx])[0], \
        np.asarray(X_data.loc[idx])[1:], raw_image_path)
    outfile = outDir +  str(idx)+ '.jpg'
    scan.save_image(outfile, width)

def do_test_train_split(filename):
    """
    Does a test train split if not previously done

    """
    candidates = pd.read_csv(filename)

    positives = candidates[candidates['class']==1].index  
    negatives = candidates[candidates['class']==0].index

    ## Under Sample Negative Indexes
    np.random.seed(42)
    negIndexes = np.random.choice(negatives, len(positives)*5, replace = False)

    candidatesDf = candidates.iloc[list(positives)+list(negIndexes)]

    X = candidatesDf.iloc[:,:-1]
    y = candidatesDf.iloc[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y,\
     test_size = 0.20, random_state = 42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, \
        test_size = 0.20, random_state = 42)

    X_train.to_pickle('traindata')
    y_train.to_pickle('trainlabels')
    X_test.to_pickle('testdata')
    y_test.to_pickle('testlabels')
    X_val.to_pickle('valdata')
    y_val.to_pickle('vallabels')

def main():
    if len(sys.argv) < 2:
        raise ValueError('1 argument needed. Specify if you need to generate a train, test or val set')
    else:
        mode = sys.argv[1]
        if mode not in ['train', 'test', 'val']:
            raise ValueError('Argument not recognized. Has to be train, test or val')

    inpfile = mode + 'data'
    outDir = mode + '/image_'

    if os.path.isfile(inpfile):
        pass
    else:
        do_test_train_split(candidates_file)
    X_data = pd.read_pickle(inpfile)
    
    [create_data(idx, outDir, X_data) for idx in X_data.index]
    
    # Parallel(n_jobs = 3)(delayed(create_data)(idx, outDir, X_data) for idx in X_data.index)

if __name__ == "__main__":
    main()