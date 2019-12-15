import copy

import numpy as np

from cv2 import cv2

from config.Config import Config


class KMeanClustring(object):

    def __init__(self, grid_without_obs):

        self._grid = copy.copy(grid_without_obs)
        self._grid_len = Config.GRID_LEN
        self._grid_width = Config.GRID_WIDTH
        self._free_space = Config.FREE_SPACE
        self._no_of_centroids = self._find_no_of_centroids()    # Function
        self._free_points = None
        self._roi = None
        self._generate_points()                                 # Function
        self.get_random_centroids()
    

    def _generate_points(self):

        width_min = self._free_space
        width_max = self._grid_width-self._free_space
        len_min = self._free_space
        len_max = self._grid_len-self._free_space

        temp_grid = copy.copy(self._grid)

        self._roi = copy.copy(temp_grid[len_min:len_max, width_min:width_max])
        temp_points = np.where(np.all(self._roi == [255, 255, 255], axis=-1))
        self._roi[temp_points] = [0,0,255]

        temp_grid = copy.copy(self._grid)

        temp_grid[len_min:len_max, width_min:width_max] = self._roi

        self._free_points = np.where(np.all(temp_grid == [0,0,255], axis=-1))

        cv2.imshow('Occupancy_grid', temp_grid)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def get_random_centroids(self):

        temp_x = np.random.choice(self._free_points[0][:], size=self._no_of_centroids)
        temp_y = np.random.choice(self._free_points[1][:], size=self._no_of_centroids)

        for i in range(len(temp_x)):
            self._grid = cv2.circle(self._grid, (temp_y[i], temp_x[i]), 5, (0,255,0), -1)

        cv2.imshow('Occupancy_grid', self._grid)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def _find_no_of_centroids(self):

        temp_len = self._grid_len - self._free_space
        temp_width = self._grid_width - self._free_space

        temp_area = temp_len * temp_width
        desired_area_of_a_region = 200*200

        return int(temp_area/desired_area_of_a_region)
