#@title MIT License
#
# Copyright (c) 2019 Andreas Eberlein
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


# # Prepare data for training of semantic segmentation network based on Virtual KITTI dataset


from __future__ import absolute_import, division, print_function, unicode_literals
import os
import numpy as np
import matplotlib.pyplot as plt

import cv2
import h5py
import pickle

import data_helper


def prepare_data_func(input_path, output_path, image_rescale_factor):
    input_folder_rgb = os.path.join(input_path, 'vkitti_1.3.1_rgb')
    output_folder_rgb = os.path.join(output_path, 'vkitti_1.3.1_rgb')
    data_helper.compressPngCopyTxt(input_folder_rgb, output_folder_rgb, image_rescale_factor)

    input_folder_seg = os.path.join(input_path, 'vkitti_1.3.1_scenegt')
    output_folder_seg = os.path.join(output_path, 'vkitti_1.3.1_scenegt')
    data_helper.compressPngCopyTxt(input_folder_seg, output_folder_seg, image_rescale_factor)

    encoding_file_path = os.path.join(output_path,
             'vkitti_1.3.1_scenegt/0001_sunset_scenegt_rgb_encoding.txt')

    # Get lists of training and validation files and read them from hard drive:
    train_image_folder = os.path.join(output_folder_rgb, '0001/sunset')
    train_image_files = os.listdir(train_image_folder)
    print("Number of training images for now: ", len(train_image_files))

    # valid_image_folder = os.path.join(output_folder_rgb, '0002/sunset')
    # valid_image_files = os.listdir(valid_image_folder)
    # print("Number of validation images for now: ", len(valid_image_files))

    train_label_folder = os.path.join(output_folder_seg, '0001/sunset')
    train_label_files = os.listdir(train_label_folder)
    print("Number of training label images for now: ", len(train_label_files))

    # valid_label_folder = os.path.join(output_folder_seg, '0002/sunset')
    # valid_label_files = os.listdir(valid_label_folder)
    # print("Number of validation images for now: ", len(valid_label_files))

    # TODO: Make preprocessing more efficient by storing the images in the hdf5 file directly
    train_images = data_helper.loadImagesToArray(train_image_folder, train_image_files)
    # valid_images = data_helper.loadImagesToArray(valid_image_folder, valid_image_files)

    train_labels = data_helper.loadImagesToArray(train_label_folder, train_label_files,
                                                 normalize=False)
    # valid_labels = data_helper.loadImagesToArray(valid_label_folder, valid_label_files,
    #                                              normalize=False)

    # Next step: Reading encoding data and convert labels to one-hot encoded array

    instance_dict = data_helper.parseCategoryFile(encoding_file_path)
    values_categories_dict, categories_ids_dict, ids_categories_dict, \
        ids_values_dict = data_helper.reduceInstancesToCategories(instance_dict)
    values_ids_dict = data_helper.oneHotEncodingDict(values_categories_dict, categories_ids_dict)

    print(categories_ids_dict)

    print(train_labels.shape)

    one_hot_encoded_labels = data_helper.oneHotEncodeImages(train_labels, values_ids_dict)

    # Note: Everytime we recompute the dictionaries that map values to categories, the order
    # of the elements in the dictionary may change. Therefore and in order to simplify further
    # processing, we store the data in an hdf5 file.
    #
    # TODO: We should sort the dictionaries according to the name of the categories. This would
    # allow to establish a stable sorting!

    processed_training_data_file_name = os.path.join(output_path,
                                                     'Preprocessed/ProcessedTrainingData.hdf5')

    data_file = h5py.File(processed_training_data_file_name, 'w')
    labels_data_set = data_file.create_dataset('one_hot_encoded_labels',
                                               one_hot_encoded_labels.shape,
                                               dtype=one_hot_encoded_labels.dtype)
    labels_data_set[...] = one_hot_encoded_labels

    images_data_set = data_file.create_dataset('training_images',
                                               train_images.shape,
                                               dtype=train_images.dtype)
    images_data_set[...] = train_images

    data_file.close()

    dict_file_name = os.path.join(output_path, 'Preprocessed/Dictionaries.dat')
    dict_file = open(dict_file_name, 'wb')
    value_category_id_mappings = [values_categories_dict, categories_ids_dict,
                                  ids_categories_dict, ids_values_dict,
                                  values_ids_dict]
    pickle.dump(value_category_id_mappings, dict_file, pickle.HIGHEST_PROTOCOL)

    dict_file.close()

    # ## TODO:
    # - Enlarge dataset by loading more folders and merging of labels files
    # - Normalize input images from pixel values of 0...255 to 0...1 (and float values)
