'''!
@file card_extractor.py
@brief Extracts Card and Boundingbox from an Image
@details Uses OpenCV to extract the part of an image corresponding
         to a card as the bounding boxes of the rank/suit in the top 
         left and bottom right
@author Julian Massing
'''

from enum import Enum
import math

import cv2 as cv
import numpy as np

class FeatureDetector:
    '''! Class to detect features in images '''
    def __init__(self, image):
        '''!
        @brief Constructor

        @param image (numpy.ndarray) The raw image
        '''
        self.image = image
    
    def detect_contours(self, threshold):
        '''!
        @brief Run contour detection

        @param threshold (int) Binarization threshold
        @return binary_img (numpy.ndarray) Binarized image
        @return contours (list) The contours in the image
        @return The contour hierarchy as a list
        '''
        gray_img = cv.cvtColor(self.image, cv.COLOR_BGR2GRAY)
        ret, binary_img = cv.threshold(gray_img, threshold, 255, cv.THRESH_BINARY)
        contours, hierarchy = cv.findContours(binary_img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        return binary_img, contours, hierarchy
    
    def detect_corners(self, contours):
        '''!
        @brief Detect corner points of Contours

        @param contours (list) Contours
        @return List of corner points of each contour  
        '''
        d = 0
        corners = []
        for contour in contours:
            points = []
            while True:
                d = d + 0.1
                points = cv.approxPolyDP(contour, d, True)
                if len(points) <= 4:
                    break
            corners.append(points)
        
        return corners
    
    def detect_centers(self, contours):
        '''!
        @brief Detect center points of Contours

        @param contours (list) Contours
        @return List of center points of each contour  
        '''

        centers = []
        for contour in contours:
            M = cv.moments(contour)
            centers.append([ int((M["m10"] / M["m00"])), int((M["m01"] / M["m00"]))])
        
        return centers

class FilterType(Enum):
    '''! Enum for filter methods '''
    larger = 1 #!< Everything larger than threshold is removed
    smaller = 2 #!< Everything smaller than threshold is removed

class ContourFilter:
    '''! Class contains methods to filter contours '''
    def filter_by_area(self, contours, threshold, method):
        '''!
        @brief Filter contours by their area
        @details Either contours with an area smaller or larger than the threshold
                 area are filtered, depending on the method.
        @param contours (list) Contours to filter
        @param threshold (int) Size of the threshold area
        @param method (Filter) One of FilterType.larger or FilterType.smaller
        @return List of filtered contours
        '''
        if(method == FilterType.larger):
            filtered = [contour for contour in contours if cv.contourArea(contour) < threshold]
        elif(method == FilterType.smaller):
            filtered = [contour for contour in contours if cv.contourArea(contour) > threshold]
        else:
            raise("Unknown Filter Method.")
        
        return filtered

class ImageDewarper:
    '''! @Brief Class to dewarp images '''

    def __init__(self):
        '''! @brief The constructor '''
        self.__origin = []

    def __angle(self, point):
        '''!
        @brief Calculate the angle between the Vector to a point in
               the coordinate system with the member __origin as the 
               center and the reference vector [1, 0]
        
        @param point (list) Point of interest
        @return Angle in radiant
        '''
        point = point[0]
        refvec = [1, 0]
        # Vector between point and the origin: v = p - o
        vector = [point[0]-self.__origin[0], point[1]-self.__origin[1]]
        # Length of vector: ||v||
        lenvector = math.hypot(vector[0], vector[1])
        # If length is zero there is no angle
        if lenvector == 0:
            return -math.pi, 0
        # Normalize vector: v/||v||
        normalized = [vector[0]/lenvector, vector[1]/lenvector]
        dotprod  = normalized[0]*refvec[0] + normalized[1]*refvec[1]     # x1*x2 + y1*y2
        diffprod = refvec[1]*normalized[0] - refvec[0]*normalized[1]     # x1*y2 - y1*x2
        angle = math.atan2(diffprod, dotprod)
        # Negative angles represent counter-clockwise angles so we need to subtract them 
        # from 2*pi (360 degrees)
        if angle < 0:
            angle += 2*math.pi
        return angle

    def __sort_corners(self, corners, center):
        '''!
        @brief Sorts corner points counterclockwise starting at upper right 
               looking at the image
        
        @param corners (list) Corner points of the image
        @param center (list) Center point of the image
        @return List of sorted points
        '''
        self.__origin = center
        return sorted(corners, key=self.__angle)

    def __transformation_coordinates(self, dsize: tuple):
        '''!
        @brief Calculates the coordinates where the image corners should be
               transformed to
        
        @param dsize (tuple) Size of destination image (width, height)
        @return List of coordinates
        '''
        offset = 0
        coords = [ [dsize[0] - offset, offset], 
                   [offset, offset],
                   [offset, dsize[1] - offset],
                   [dsize[0] - offset, dsize[1] - offset] ]
        return coords

    def dewarp_image(self, image, corners, center, dsize: tuple):
        '''!
        @brief Dewarp part of a given image that is enclosed by given
               corner points with the given center point to Birdseye view

        @param image (numpy.ndarray) Source image
        @param corners (list) Corner points
        @param center (list) Center point
        @param dsize (tuple) Destination size (width, height)
        @return Dewarped image (numpy.ndarray) 
        '''
        if(len(corners) != 4):
            return None

        corners = self.__sort_corners(corners, center)
        coords = self.__transformation_coordinates(dsize)

        M = cv.getPerspectiveTransform(np.array(corners, dtype = "float32"), np.array(coords, dtype = "float32"))
        transformed_image = cv.warpPerspective(image, M, dsize)

        return transformed_image

if __name__ == "__main__":
    
    image = cv.imread("./live_image.jpg")
    feature_detector = FeatureDetector(image)
    binary_img, contours, hierarchy = feature_detector.detect_contours(180)

    contour_filter = ContourFilter()
    contours = contour_filter.filter_by_area(contours, 10000, FilterType.smaller)

    corners = feature_detector.detect_corners(contours)
    centers = feature_detector.detect_centers(contours)

    dewarper = ImageDewarper()
    dewarped_image = dewarper.dewarp_image(image, corners[2], centers[2], (180, 300))
    
    cv.drawContours(image, contours, -1, (0, 255, 0), 3)

    for points in corners:
        for point in points:
            image = cv.circle(image, (point[0][0], point[0][1]), radius=3, color=(0, 0, 0), thickness=-1)
    
    for point in centers:
        image = cv.circle(image, (point[0], point[1]), radius=3, color=(192, 0, 0), thickness=-1)

    cv.imshow("Image", image)
    cv.imshow("Dewarped Card", dewarped_image)
    #cv.imwrite("AS.jpg", dewarped_image)
    k = cv.waitKey(0)