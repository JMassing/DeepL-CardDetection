'''!
@file show_train_data.py
@brief Script to visualize train images with bounding boxes
@author Julian Massing
'''

import os
import argparse 

import cv2 as cv
import pickle
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

parser = argparse.ArgumentParser(description='Visualize synthetic training data for card detection NN generated with data_generator.py.')
parser.add_argument('nr', type=int, help=''' Nr of image to visualize. Data was saved with suffix _<nr>.zfill(5). 
                     So nr = 1 will load image_00001.jpg, image_00001.pkl''')
parser.add_argument('-d', '--data', type=str, default="train_data", help='Path to data')

args = parser.parse_args()

nr = args.nr
if(os.path.exists(args.data) is False):
    print(f"[Warning]: Path {args.data} was not found")
else:
    img_file = os.path.join(args.data, f"image_{str(nr).zfill(5)}.jpg")
    annotations_file = os.path.join(args.data, f"image_{str(nr).zfill(5)}.pkl")
    image = cv.imread(img_file)
    with open(annotations_file, 'rb') as filename:
        annotations = pickle.load( filename )
    bounding_boxes = annotations["bounding_boxes"]
    bbs = [ BoundingBox( x1 = bb["x1"], x2 = bb["x2"], y1 = bb["y1"], y2 = bb["y2"],
                        label = bb["label"]) for bb in bounding_boxes ]                     
    for bb in bbs:
        image = bb.draw_on_image(image, size=2, color=[255, 0, 0])

    cv.imshow("Image", image)
    k = cv.waitKey(0)