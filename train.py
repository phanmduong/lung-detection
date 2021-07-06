#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 24 10:46:10 2021

@author: phanmduong
"""
from read_data import getData
from keras.optimizers import Adam
from RoiPooling3D import RoiPoolingConv
from keras.layers import Dense,Conv3D,MaxPooling3D,Flatten, TimeDistributed, Dropout, Input
from keras.models import Model
import losses
import math
import random
import pprint
import time
import numpy as np
from keras.utils import generic_utils

def vgg16(input):
    # """13th layer"""
    # ###### Determine proper input shape: theano or tensorflow #####
    # if K.image_dim_ordering() == 'th':
    #     input_shape = (3, None, None)
    # else:
    #     input_shape = (None, None, 3)

    # if K.image_dim_ordering() == 'tf':
    #     bn_axis = 3
    # else:
    #     bn_axis = 1

    ###### Block 1 ######
    x = Conv3D(64, (3, 3, 3), activation='relu', padding='same', name='block1_conv1')(input)
    x = Conv3D(64, (3, 3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), name='block1_pool')(x)

    ###### Block 2 ######
    x = Conv3D(128, (3, 3,3 ), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv3D(128, (3, 3,3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling3D((2, 2,2), strides=(2, 2, 2), name='block2_pool')(x)

    ###### Block 3 ######
    x = Conv3D(256, (3, 3,3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv3D(256, (3, 3,3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv3D(256, (3, 3,3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling3D((2, 2,2), strides=(2, 2, 2), name='block3_pool')(x)

    ###### Block 4 ######
    x = Conv3D(512, (3, 3,3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv3D(512, (3, 3,3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv3D(512, (3, 3,3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling3D((2, 2,2), strides=(2, 2, 2), name='block4_pool')(x)

    ###### Block 5 ######
    x = Conv3D(512, (3, 3,3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv3D(512, (3, 3,3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv3D(512, (3, 3,3), activation='relu', padding='same', name='block5_conv3')(x)

    return x

def rpn(base_layers, num_anchors):
    """
    Used for testing
    inputs:
        base_layers: feature map, output from shared CNN (raw feature extraction net)
        num_anchors: in faster rcnn, equals to 3 x 3
    output: 
        x_cls: 9-d, used to build cls_loss, cross entropy
        x_regr: 4x9-d, used to build regr_loss, smooth L1
    """
    x = Conv3D(512, (3, 3, 3), padding='same', activation='relu', kernel_initializer='normal', name='rpn_conv1')(base_layers)
    x_cls = Conv3D(num_anchors, (1, 1, 1), activation='sigmoid', kernel_initializer='uniform', name='rpn_out_class')(x)
    x_regr = Conv3D(num_anchors * 4, (1, 1,1), activation='linear', kernel_initializer='zero', name='rpn_out_regress')(x)
    return [x_cls, x_regr, base_layers]

def classifier(base_layers, input_rois, num_rois, nb_classes = 21, trainable=False):

    # compile times on theano tend to be very high, so we use smaller ROI `pooling regions to workaround

    pooling_regions = 7
    # input_shape = (num_rois,7,7,7,128)

    out_roi_pool = RoiPoolingConv(pooling_regions, num_rois)([base_layers, input_rois])

    out = TimeDistributed(Flatten(name='flatten'))(out_roi_pool)
    out = TimeDistributed(Dense(4096, activation='relu', name='fc1'))(out)
    out = TimeDistributed(Dropout(0.5))(out)
    out = TimeDistributed(Dense(4096, activation='relu', name='fc2'))(out)
    out = TimeDistributed(Dropout(0.5))(out)

    out_class = TimeDistributed(Dense(nb_classes, activation='softmax', kernel_initializer='zero'), name='dense_class_{}'.format(nb_classes))(out)
    # note: no regression target for bg class
    out_regr = TimeDistributed(Dense(4 * (nb_classes-1), activation='linear', kernel_initializer='zero'), name='dense_regress_{}'.format(nb_classes))(out)

    return [out_class, out_regr]

train_imgs, classes_count, class_mapping = getData("./data_preprocessed/augmented_meta2.csv")

if 'bg' not in classes_count:
	classes_count['bg'] = 0
	class_mapping['bg'] = len(class_mapping)

random.shuffle(train_imgs)

num_imgs = len(train_imgs)

print(train_imgs)

print('Training images per class:')
pprint.pprint(classes_count)
print(f'Num classes (including bg) = {len(classes_count)}')

print(f'Num train samples {len(train_imgs)}')

input_shape_img = (None, None, None, 3)

image_input = Input(shape=input_shape_img)
roi_input = Input(shape=(None, 6))
nb_classes = 2;
"""
C: config defined before
image_input: image data into shared CNN
roi_input: roi input, impact on the feature map
nn are function name
"""
# First part: shared CNN, could be vgg16, resnet50, inception
shared_CNN = vgg16(image_input)
# add RPN part
num_anchors = len([8, 16, 32]) * len([[1, 1], [1./math.sqrt(2), 2./math.sqrt(2)], [2./math.sqrt(2), 1./math.sqrt(2)]])
rpn = rpn(shared_CNN, num_anchors)
# add classifier (fast RCNN) part

classifier = classifier(shared_CNN, roi_input, 4, nb_classes=nb_classes, trainable=True)

model_rpn = Model(image_input,rpn[:2])
model_classifier = Model([image_input, roi_input], classifier)
# # combine two model
model_all = Model([image_input, roi_input], rpn[:2] + classifier)

# model_rpn.summary();
# model_classifier.summary();
# model_all.summary();

optimizer = Adam(lr=1e-5)
optimizer_classifier = Adam(lr=1e-5)
model_rpn.compile(optimizer=optimizer, loss=[losses.rpn_loss_cls(num_anchors), losses.rpn_loss_regr(num_anchors)])
model_classifier.compile(optimizer=optimizer_classifier, loss=[losses.class_loss_cls, losses.class_loss_regr(nb_classes-1)], metrics={f'dense_class_{nb_classes}': 'accuracy'})
model_all.compile(optimizer='sgd', loss='mae')


epoch_length = 1000
num_epochs = 10
iter_num = 0

losses = np.zeros((epoch_length, 5))
rpn_accuracy_rpn_monitor = []
rpn_accuracy_for_epoch = []
start_time = time.time()

best_loss = np.Inf

class_mapping_inv = {v: k for k, v in class_mapping.items()}
print('Starting training')

verbose = True



# for epoch_num in range(num_epochs):

#     progbar = generic_utils.Progbar(epoch_length)
#     print(f'Epoch {epoch_num + 1}/{num_epochs}')
    
#     while True:
#         try:

#             if len(rpn_accuracy_rpn_monitor) == epoch_length and verbose:
#                 mean_overlapping_bboxes = float(sum(rpn_accuracy_rpn_monitor))/len(rpn_accuracy_rpn_monitor)
#                 rpn_accuracy_rpn_monitor = []
#                 print(f'Average number of overlapping bounding boxes from RPN = {mean_overlapping_bboxes} for {epoch_length} previous iterations')
#                 if mean_overlapping_bboxes == 0:
#                     print('RPN is not producing bounding boxes that overlap the ground truth boxes. Check RPN settings or keep training.')
            
#             iter_num += 1

#             if iter_num == epoch_length:
#                 iter_num = 0
#                 break

#         except Exception as e:
#             print(f'Exception: {e}')
#             continue

# print('Training complete, exiting.')
