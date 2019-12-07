#!usr/bin/env python

# import copy
import math

# import numpy as np

# from cv2 import cv2

def l2_distance(coord1_x, coord1_y, coord2_x, coord2_y):
    '''
    Calculates the l2 distance between two world coordinates.
    For detailed explanation, visit this link:
    https://en.wikipedia.org/wiki/Euclidean_distance
    Parameters
    ----------
    coord1_x : float
        x of the first coordinate.
    coord1_y : float
        y of the first coordinate.
    coord2_x : float
        x of the second coordinate.
    coord2_y : float
        y of the second coordinate.
    Returns
    -------
    float
        l2 distance between the two given coordinates.
    '''

    dist_x = coord2_x - coord1_x
    dist_y = coord2_y - coord1_y

    return math.sqrt((dist_x * dist_x) + (dist_y * dist_y))# * 1.113195e5