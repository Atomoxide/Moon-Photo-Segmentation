import tensorflow as tf
import cv2
import os
import scipy
import numpy as np
from config import image_shape, train_dir, train_label_dir, test_dir


train_list_dir = os.listdir(train_dir)
train_list_dir.sort()
train_label_list_dir = os.listdir(train_label_dir)
train_label_list_dir.sort()

train_filenames = [train_dir + filename for filename in train_list_dir]
train_label_filenames = [train_label_dir +
                         filename for filename in train_label_list_dir]

#print(train_label_filenames)
test_list_dir = os.listdir(test_dir)
test_list_dir.sort()
test_filenames = [test_dir + filename for filename in test_list_dir]


def train_generator():
    for train_file_name, train_label_filename in zip(train_filenames, train_label_filenames):
        image, label = handle_data(train_file_name, train_label_filename)

        yield tf.convert_to_tensor(image), tf.convert_to_tensor(label)


def test_generator():
    for test_filename in test_filenames:
        image = handle_data(test_filename)

        yield tf.convert_to_tensor(image)


def handle_data(train_filenames, train_label_filenames=None):
    image = scipy.misc.imresize(
        scipy.misc.imread(train_filenames), image_shape)
    
    image_yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    image_yuv[:, :, 0] = cv2.equalizeHist(image_yuv[:, :, 0])
    image = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2RGB)

    if train_label_filenames is not None:
        #print('FOUND LABEL FILE')
        gt_image = scipy.misc.imresize(
            scipy.misc.imread(train_label_filenames), image_shape)

        # In case some color degrade or demolition
        gt_image = np.where(gt_image!=0,255,gt_image)
        
        #Define four labels
        background_color = np.array([255, 0, 0])
        ground_color = np.array([0,0,0])
        big_rock_color = np.array([0,0,255])
        small_rock_color = np.array([0,255,0])

        gt_bg = np.all(gt_image == background_color, axis=2)
        gt_g = np.all(gt_image == ground_color, axis=2)
        gt_br = np.all(gt_image == big_rock_color, axis=2)
        gt_sr = np.all(gt_image == small_rock_color, axis=2)

        gt_bg = gt_bg.reshape(*gt_bg.shape, 1)
        gt_g = gt_bg.reshape(*gt_g.shape, 1)
        gt_br = gt_bg.reshape(*gt_br.shape, 1)
        gt_sr = gt_bg.reshape(*gt_sr.shape, 1)

        gt_image = np.concatenate((gt_bg, gt_g, gt_br,gt_sr), axis=2)
    
        return np.array(image), gt_image
    else:
        return np.array(image)


