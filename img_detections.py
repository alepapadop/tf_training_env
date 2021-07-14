import os
import subprocess
import sys
import requests
import glob
import tempfile
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
import warnings
from zipfile import ZipFile
from git import Repo, RemoteProgress
from tqdm import tqdm
from shutil import copyfile
from argparse import ArgumentParser
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import tensorflow as tf

label_map_path = os.path.join('workspace', 'training_demo', 'annotations', 'label_map.pbtxt')
category_index = ""
detect_fn = ""

def LoadModel(model_dir_name):
    global label_map_path
    global category_index
    global detect_fn

    exported_model_path = os.path.join('workspace', 'training_demo', 'exported_models', model_dir_name, 'saved_model')

    if not os.path.exists(exported_model_path):
        exit('The directory: ' + exported_model_path + ' does not exists')

    detect_fn = tf.saved_model.load(exported_model_path)
    category_index = label_map_util.create_category_index_from_labelmap(label_map_path, use_display_name = True)




def LoadImages(img_dir, threshold):

    if not os.path.exists(img_dir):
        exit('Image directory: ' + img_dir + ' does not exists')

    images = []
    img_ext = ['.jpg', '.png']
    for img in os.listdir(img_dir):
        ext = os.path.splitext(img)[1]
        if ext in img_ext:
            images.append(img)

    matplotlib.use('Qt5Agg')
    for img in images:
        img_path = os.path.join(img_dir, img)
        image_np = np.array(Image.open(img_path))

        input_tensor = tf.convert_to_tensor(image_np)
        # The model expects a batch of images, so add an axis with `tf.newaxis`.
        input_tensor = input_tensor[tf.newaxis, ...]
    
        # input_tensor = np.expand_dims(image_np, 0)
        detections = detect_fn(input_tensor)
    
        # All outputs are batches tensors.
        # Convert to numpy arrays, and take index [0] to remove the batch dimension.
        # We're only interested in the first num_detections.
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                       for key, value in detections.items()}
        detections['num_detections'] = num_detections

        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
    
        image_np_with_detections = image_np.copy()
    
        viz_utils.visualize_boxes_and_labels_on_image_array(
              image_np_with_detections,
              detections['detection_boxes'],
              detections['detection_classes'],
              detections['detection_scores'],
              category_index,
              use_normalized_coordinates = True,
              max_boxes_to_draw = 200,
              min_score_thresh = 0.3,
              agnostic_mode = False)
        
        plt.figure()
        plt.imshow(image_np_with_detections)
        print('Done')

    plt.show()

def main():

    parser = argparse.ArgumentParser(description = "Test the model on image",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-n', '--exported_model_dir_name',
        help = 'Name of an existing directory containing the trained model.',
        type = str,
        default = ""
    )

    parser.add_argument(
        '-d', '--image_dir_path',
        help = 'Name of an existing directory containing images.',
        type = str,
        default = ""
    )

    parser.add_argument(
        '-t', '--threshold',
        help = 'Threshold.',
        type = float,
        default = 0.3
    )

    args = parser.parse_args()

    if not args.exported_model_dir_name:
        exit('Please provide the model name')

    if not args.image_dir_path:
        exit('Please provide an image directory path')



    LoadModel(args.exported_model_dir_name)
    LoadImages(args.image_dir_path, args.threshold)


if __name__ == "__main__":
    main()
