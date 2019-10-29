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

import file_helper
import shutil
import os
import cv2
import itertools
import numpy as np
import matplotlib.pyplot as plt


def plot_as_rgb_image(image, input_normalized=False):
    """ Plot BGR encoded image as RGB image

    Parameters
    ----------
    image: Numpy array
            Image data in BGR format
    input_normalized: bool
            Flag indicating whether the input image is normalized to [-1, 1]

    Note
    ----
    plt.show() has to be called afterwards

    Returns
    -------
    None
    """
    if input_normalized:
        image = np.array((image + 1) * 128, dtype=np.uint8)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def compressPngCopyTxt(input_folder_name, output_folder_name, image_rescale_factor):
    """Compress png images and copy them together with text files to new folder

    Parameters
    ----------
    input_folder_name: string
            Name of input folder
    output_folder_name: string
            Name of output folder
    image_rescale_factor: float
            Factor for rescaling image dimensions
    
    Returns
    -------
    None
    """
    file_list_input = file_helper.getFileListInclSubfolders(input_folder_name)
    file_list_output = []

    for in_file_path in file_list_input:
        out_file_path = in_file_path.replace(input_folder_name, output_folder_name)
        file_list_output.append(out_file_path)
        
        if out_file_path[-4:] == '.txt':
            last_slash_pos = out_file_path.rfind('/')
            os.makedirs(os.path.dirname(out_file_path), exist_ok=True)
            shutil.copy(in_file_path, out_file_path[:last_slash_pos])
            
        elif out_file_path[-4:] == '.png':
            image = cv2.imread(in_file_path, cv2.IMREAD_UNCHANGED)
            new_size = (image.shape[1] // image_rescale_factor,
                        image.shape[0] // image_rescale_factor)
            resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_CUBIC)
            os.makedirs(os.path.dirname(out_file_path), exist_ok=True)
            cv2.imwrite(out_file_path, resized_image)


def parseCategoryFile(file_name):
    """Parse vKITTI category file and return dictionary of instance:colour pairs

    Parameters
    ----------
    file_name: string
            Name of category file

    Returns
    -------
    instance_value_dict: dict
            Dictionary mapping instance ids to colour values
    """
    instance_value_dict = {}
    with open(file_name, 'r') as file:
        for line in file:
            split_line = line.split(' ')
            if split_line[1].isdigit():
                instance_value_dict[split_line[0]] = \
                    (int(split_line[1]), int(split_line[2]), int(split_line[3]))
    return instance_value_dict


def reduceInstancesToCategories(instance_value_dict):
    """Compile dictionaries for mapping between values, object categories and ids

    Parameters
    ----------
    instance_value_dict: dict
            Dictionary mapping instance ids to colour values

    Returns
    -------
    values_categories: dict
            Dictionary mapping values to categories
    categories_ids: dict
            Dictionary mapping categories or object class names to object class ids
    ids_categories: dict
            Dictionary mapping object class ids to categories (or object class names)
    ids_vlaues: dict
            Dictionary mapping object class ids to colour values
    """
    values_categories = {}
    for item in instance_value_dict.items():
        if ':' not in item[0]:
            values_categories[item[1]] = item[0]
        else:
            pos = item[0].find(':')
            values_categories[item[1]] = item[0][0:pos]

    categories = set(values_categories.values())

    categories_ids = {category: id for id, category in zip(itertools.count(), categories)}
    ids_categories = {item[1]: item[0] for item in categories_ids.items()}

    ids_values = {}
    for item in categories_ids.items():
        idx = list(values_categories.values()).index(item[0])
        values = list(values_categories.keys())[idx]
        ids_values[item[1]] = values

    return values_categories, categories_ids, ids_categories, ids_values


def oneHotEncodingDict(values_categories, categories_ids):
    """Compile dictionaries for mapping colour values to object class ids

    Parameters
    ----------
    values_categories: dict
            Dictionary mapping colour values to object class names
    categories_ids: dict
            Dictionary mapping object class names to object class ids

    Returns
    -------
    values_ids: dict
            Dictionary mapping colour values to object class ids
    """
    values_ids = {}

    for item in values_categories.items():
        values_ids[item[0]] = categories_ids[item[1]]

    return values_ids


def loadImagesToArray(path, file_list, normalize=True):
    """Load images from hard drive to numpy array

    Parameters
    ----------
    path: string
            Name of folder where files to be read are located
    file_list: list
            List of file names
    normalize: bool
            Flag indicating whether images should be normalized to [-1, 1]

    Returns
    -------
    image_data: Numpy array
            Array containing image data
    """
    image_data = []
    for file_name in file_list:
        image = cv2.imread(os.path.join(path, file_name), cv2.IMREAD_UNCHANGED)
        if (normalize):
            image = image / 128 - 1
        image_data.append(image)

    return np.array(image_data)


def oneHotEncodeImages(train_labels, values_ids_dict):
    """Transform images with semantic classes encoded as colour into one-hot encoded images

    Parameters
    ----------
    train_labels: Numpy array
            Array containing all images with semantic classes encoded as colours
    values_ids_dict: dict
            Dictionary mapping colour values to object class ids

    Returns
    -------
    one_hot_encoded_images: Numpy array
            Array containing semantic information of images with one-hot encoding of classes
    """
    one_hot_encoding_dim = len(set(values_ids_dict.values()))
    one_hot_encoded_images = np.zeros(shape=(train_labels.shape[0],
                                             train_labels.shape[1],
                                             train_labels.shape[2],
                                             one_hot_encoding_dim), dtype=np.uint8)

    for i in range(train_labels.shape[0]):
        for j in range(train_labels.shape[1]):
            for l in range(train_labels.shape[2]):
                # Note: OpenCV stores pictures in BGR format, 
                # so we need to flip the colors for comparisons!
                bgr_pixel_val = tuple(train_labels[i][j][l])
                rgb_pixel_val = (bgr_pixel_val[2], bgr_pixel_val[1], bgr_pixel_val[0])
                id = values_ids_dict[rgb_pixel_val]
                values = np.array([int(i == id) for i in range(one_hot_encoding_dim)])
                one_hot_encoded_images[i][j][l] = values

    return one_hot_encoded_images


def oneHotDecodeImages(one_hot_encoded_labels, ids_values_dict):
    """Transform one-hot encoded images with semantic classes to colour encoded images

    Parameters
    ----------
    one_hot_encoded_labels: Numpy array
            Array containing all images with semantic classes in one-hot encoding
    ids_values_dict: dict
            Dictionary mapping object class ids to colour values

    Returns
    -------
    colour_coded_labels: Numpy array
            Array containing semantic information of images with colour encoding of classes
    """
    colour_coded_labels = np.zeros(shape=(one_hot_encoded_labels.shape[0],
                                          one_hot_encoded_labels.shape[1],
                                          one_hot_encoded_labels.shape[2], 3), dtype=np.uint8)

    for i in range(one_hot_encoded_labels.shape[0]):
        for j in range(one_hot_encoded_labels.shape[1]):
            for l in range(one_hot_encoded_labels.shape[2]):
                category_id = np.argmax(one_hot_encoded_labels[i][j][l])
                # Note: OpenCV stores pictures in BGR format,
                # so we need to flip the colors for comparisons!
                rgb_pixel_val = ids_values_dict[category_id]
                bgr_pixel_val = tuple([rgb_pixel_val[2], rgb_pixel_val[1], rgb_pixel_val[0]])
                colour_coded_labels[i][j][l] = bgr_pixel_val

    return colour_coded_labels
