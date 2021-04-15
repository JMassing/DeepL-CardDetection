'''!
@file data_generator.py
@brief Generate training data (images, bounding boxes and labels) for card detection NN
@details Uses Describable Texture Dataset for Background https://www.robots.ox.ac.uk/~vgg/data/dtd/
@author Julian Massing
'''
import argparse
import os
import random

import cv2 as cv
import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from shapely.geometry import Polygon
import pickle

from card_extractor import FeatureDetector, ContourFilter, FilterType 

class DataGenerator:
    '''! 
    @brief Generates training data, i.e synthetic images with a size of 640x480 pixel, 
    bounding boxes and labels 
    '''
    def __init__(self, pad_borders=(0, 0, 0, 0), pad_val=[0, 0, 0]):
        '''!
        @brief Constructor

        @param pad_borders (tuple) Size of image padding borders (top, bottom, left, right)
        @param pad_val (list) BGR value of image padding color
        ''' 
        self.__card_size = (100, 140)
        self.__background_size = (640, 480)
        self.__pad_val = pad_val
        self.__pad_borders = pad_borders
        self.__background = None
        self.__train_image = None
        self.__cards = []
        self.__augmented = []
        self.__bounding_boxes = []
        self.__labels = []

    @property
    def background(self):
        return self.__background

    @background.setter 
    def background(self, image):
        self.__background = cv.resize(src=image, dsize=self.__background_size)
    
    @property
    def cards(self):
        return self.__cards

    @cards.setter 
    def cards(self, cards):
        assert(isinstance(cards, list))
        self.__cards = [cv.resize(src=image, dsize=self.__card_size) for image in cards]
    
    @property
    def labels(self):
        return self.__labels

    @labels.setter 
    def labels(self, labels):
        self.__labels = labels

    @property
    def augmented(self):
        return self.__augmented
    
    @property
    def train_image(self):
        return self.__train_image
    
    @train_image.setter
    def train_image(self, image):
        self.__train_image = image
    
    @property
    def bounding_boxes(self):
        return self.__bounding_boxes

    def choose_random_background(self, path):
        '''!
        @brief Chooses a random background image from all image files 
               at the given path
        
        @path Path to image files
        '''
        if(os.path.exists(path) is False):
            print(f"[Warning]: Path {path} was not found")
        else:
            _, _, filenames = next(os.walk(path))
            idx = random.randint(0, len(filenames)-1)
            image_file = os.path.join(path, filenames[idx])
            self.background = cv.imread(image_file)
    
        return

    def __get_convex_hulls(self, image, threshold):
        '''!
        @brief Get the combined convex hull of the rank and suit 

        @param image (numpy.ndarray) Image containing rank and suit
        @param threshold (int) Binarization threshold
        @return Convex hull (list of points)
        '''
        proc = FeatureDetector(image)
        _, contours, _ = proc.detect_contours(threshold)
        list_of_pts = []
        # combine the contours of rank and suit
        for contour in contours[1:]:
            list_of_pts += [pt[0] for pt in contour]
            contour = np.array(list_of_pts).reshape((-1,1,2)).astype(np.int32)
        return cv.convexHull(contour)

    def __pad_image(self, image):
        '''!
        @brief Padds images 

        @param image (numpy.ndarray) image to padd
        @return Padded image (numpy.ndarray) 
        '''
        return cv.copyMakeBorder(
            src=image, top=self.__pad_borders[0], bottom=self.__pad_borders[1], 
            left=self.__pad_borders[2], right=self.__pad_borders[3], 
            borderType=cv.BORDER_CONSTANT, value=self.__pad_val )
    
    def __black_image(self, dsize: tuple):
        '''!
        @brief Create a black image of dtype uint8

        @param dsize Image size
        @return Black image (dtype = 'uint8')
        '''
        return np.zeros(dsize, dtype='uint8')
    
    def __get_overlap(self, poly1, poly2):
        '''! 
        @brief Gets the intersection area of two polygons

        @param poly1 (shapely.polygon) First polygon
        @param poly2 (shapely.polygon) Second polygon
        @return Overlap (float)
        '''
        return poly1.intersection(poly2).area
    
    def __generate_random_mask(self):
        '''!
        @brief Generates mask image with black background and cards
               placed at random positions in the image. Cards may 
               overlap. Bounding boxes are shifted to card positions.
            
        @return Mask image (numpy.ndarray with shape = self.__background.shape, dtype = uint8)
        '''
        mask_image = self.__black_image((self.__background.shape[0], self.__background.shape[1] , 3))
        # add cards to mask image
        for i, card in enumerate(self.__augmented):
            # choose random postion for card on mask image
            pos_start = ( random.randint(0, mask_image.shape[0] - card.shape[0]), 
                          random.randint(0, mask_image.shape[1] - card.shape[1]) )
            pos_end = (pos_start[0]+card.shape[0], pos_start[1]+card.shape[1])
            # Add card to mask
            mask_clipping = mask_image[pos_start[0]:pos_end[0], pos_start[1]:pos_end[1]]
            mask_clipping = np.where(card, card, mask_clipping)
            mask_image[pos_start[0]:pos_end[0], pos_start[1]:pos_end[1]] = mask_clipping
            # Add shift to bb
            for j, bb in enumerate(self.__bounding_boxes[i].bounding_boxes):
                self.__bounding_boxes[i].bounding_boxes[j] = \
                    bb.shift(x = pos_start[1], y = pos_start[0])

            #delete bounding boxes from other cards if card overlaps them
            if i>0:             
                # Get corners of card    
                proc = FeatureDetector(card)
                _, contours, _ = proc.detect_contours(140)
                contour_filter = ContourFilter()
                contours = contour_filter.filter_by_area(contours, 5000, FilterType.smaller)
                corners = proc.detect_corners(contours)
                corners = np.reshape(corners, (4, 2)) + [pos_start[1], pos_start[0]]
                p1 = Polygon( [(corners[0][0], corners[0][1]), (corners[1][0], corners[1][1]), 
                              (corners[2][0], corners[2][1]), (corners[3][0], corners[3][1])] )

                # Get overlap for bbs of all cards already on the image
                for k in range(i):
                    idx = 0
                    while True:
                        bb = self.__bounding_boxes[k].bounding_boxes[idx]
                        p2 = Polygon( [(bb.x1_int, bb.y1_int), (bb.x2_int, bb.y1_int), 
                                      (bb.x2_int, bb.y2_int), (bb.x1_int, bb.y2_int)] )
                        # delete bb if overlap is > 20%
                        if(self.__get_overlap(p1, p2)/p2.area*100 > 20):
                            self.__bounding_boxes[k].bounding_boxes.pop(idx)
                            # Lower index by 1, since pop removes an element of the list
                            idx -= 1
                        idx += 1
                        if idx == len(self.__bounding_boxes[k].bounding_boxes):
                            break

        return mask_image

    def pick_random_cards(self, path, num_cards):
        '''!
        @brief Randomly picks <num_cards> card images from images in path.

        @param path (str) Path to image files
        @param num_cards (int) Number of card images to pick
        '''
        cards = []
        labels = []
        if(os.path.exists(path) is False):
            print("[Warning]: Path was not found")
        else:
            _, _, filenames = next(os.walk(path))
            idxs = random.sample(range(0, len(filenames)), num_cards)
            for idx in idxs:
                image = os.path.join(path, filenames[idx])
                cards.append(cv.imread(image))
                labels.append(os.path.splitext(filenames[idx])[0])
        self.cards = cards
        self.labels = labels
        
        return 

    def generate_data(self):
        '''! Generates training image '''

        for i, card in enumerate(self.__cards):
            # Get Bounding Boxes for Rank/Suit image in top left and bottom
            # right corners of the card. 
            top = self.__black_image((card.shape[0], card.shape[1], 3)) 
            bottom = top.copy()
            # height and width of corners
            height = int(card.shape[0]/2.5)
            width = int(card.shape[1]/5)
            top[:height, :width] = card[:height, :width]
            bottom[-height:, -width:] = card[-height:, -width:]
            top_hull = self.__get_convex_hulls(top, 140)
            bottom_hull = self.__get_convex_hulls(bottom, 140)
            bb_top = np.asarray(cv.boundingRect(top_hull), dtype='int32')
            bb_bottom = np.asarray(cv.boundingRect(bottom_hull), dtype='int32')

            # Randomly change brightness and contrast of cards
            seq = iaa.Sequential([
                iaa.Multiply((0.85, 1.15)),
                iaa.LinearContrast((0.85, 1.15))
            ], random_order=True) 
            card = seq(image = card)

            # Add padding to images and bounding boxes
            card = self.__pad_image(card)
            bb_top[0:2] += [self.__pad_borders[2], self.__pad_borders[0]]
            bb_bottom[0:2] += [self.__pad_borders[2], self.__pad_borders[0]]

            # Transform images and Bounding boxes
            bbs = BoundingBoxesOnImage([
                BoundingBox( x1 = bb_top[0], x2 = bb_top[0] + bb_top[2], 
                             y1 = bb_top[1], y2 = bb_top[1] + bb_top[3],
                             label = self.__labels[i] ),
                BoundingBox( x1 = bb_bottom[0], x2 = bb_bottom[0] + bb_bottom[2], 
                             y1 = bb_bottom[1], y2 = bb_bottom[1] + bb_bottom[3],
                             label = self.__labels[i] )
            ], shape=card.shape)
            # Randomly apply affine transformations
            seq = iaa.Sequential([
                iaa.Affine(
                    scale=(0.75, 1.25), 
                    rotate=(-90, 90),
                    shear = (-8, 8),
                )
            ], random_order=True) 

            card_aug, bbs_aug = seq(image=card, bounding_boxes=bbs)
            self.__augmented.append(card_aug)
            self.__bounding_boxes.append(bbs_aug)
                 
        # Combine mask and background image to training image
        mask = self.__generate_random_mask()

        self.__train_image = np.where(mask, mask, self.__background)
       
        return

    def write_data(self, path):
        '''! 
        @brief Save training images as .jpg and pickle annotations 
        
        @param path (str) Path to save data
        '''
        # filenames
        img_file = f"image_{str(nr+1).zfill(5)}.jpg"
        bb_file = f"image_{str(nr+1).zfill(5)}.pkl"

        # convert bounding boxes to annotation dict and pickle
        bb = []
        for bbs in generator.bounding_boxes:
            bb += [ {"x1" : bb.x1_int, "y1": bb.y1_int, "x2": bb.x2_int, 
                     "y2": bb.y2_int, "label": bb.label} for bb in bbs ]
        annotation = {"image": img_file, "bounding_boxes": bb}
        with open(os.path.join(path, bb_file), 'wb') as output:
            pickle.dump(annotation, output, -1)
        
        # save training image
        cv.imwrite(os.path.join(path, img_file), self.__train_image)

    def clear(self):
        '''! Clear all images, bounding boxes and labels '''
        self.__background = None
        self.__train_image = None
        self.__cards = []
        self.__augmented = []
        self.__bounding_boxes = []
        self.__labels = []


parser = argparse.ArgumentParser(description='Generate synthetic training data for card detection NN.')
parser.add_argument('-c', '--cards', type=str, default="cards", help='Path to card images')
parser.add_argument('-b', '--backgrounds', type=str, default="dtd/images", help='Path to Background images')
parser.add_argument('-d', '--data', type=str, default="train_data", help='Where to save training data after generation')
parser.add_argument('-nc', '--num_cards', type=int, default=2, help='Number of cards present in image')
parser.add_argument('-ni', '--num_images', type=int, default=1000, help='Number of Synthetic images to generate')

args = parser.parse_args()

if __name__ == "__main__":

    generator = DataGenerator(pad_borders=(40, 40, 80, 80))
    nr = 0
    while nr < args.num_images:
        generator.choose_random_background(args.backgrounds)
        generator.pick_random_cards(args.cards, args.num_cards)
        generator.generate_data()
        generator.write_data(args.data)
        generator.clear()
        nr += 1  
   