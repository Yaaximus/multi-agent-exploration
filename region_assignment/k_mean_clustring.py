import copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from cv2 import cv2
from config.Config import Config
from sklearn.cluster import KMeans


class KMeanClustring(object):

    def __init__(self, grid_without_obs):

        self._roi = None
        self._colmap = []
        self._centroids = None
        self._free_points = None
        self._regions_xy_points = []
        self._grid_with_regions = None        
        self._grid_len = Config.GRID_LEN
        self._grid_width = Config.GRID_WIDTH
        self._free_space = Config.FREE_SPACE
        self._grid = copy.copy(grid_without_obs)
        self._no_of_centroids = Config.NO_OF_AGENTS
        
        self._generate_points()
        self._generate_color_map()
        # self._get_random_centroids()
        # self._no_of_centroids = self._find_no_of_centroids()


    def show_regions_with_centroids(self):

        temp_grid = copy.copy(self._grid_with_regions)

        for idx, centroid in enumerate(self._centroids):
            temp_grid = cv2.circle(temp_grid, (int(centroid[0]), int(centroid[1])), 5, (0,255,0), -1)

        cv2.imshow('OCCUPANCY GRID WITH REGIONS AND CENTROIDS', temp_grid)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def show_regions(self):

        temp_grid = copy.copy(self._grid_with_regions)

        cv2.imshow('OCCUPANCY GRID WITH REGIONS', temp_grid)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    
    def get_grid_with_regions(self):

        return self._grid_with_regions


    def find_regions(self):

        self._find_grid_with_regions()
        self._find_all_regions_xy_points()


    def get_regions_xy_points(self):

        return self._regions_xy_points


    def get_centroids(self):

        return self._centroids


    def get_color_map(self):

        return self._colmap


    def _generate_color_map(self):

        for _ in range(self._no_of_centroids):

            temp_bgr = list(np.random.randint(0,255,3))
            self._colmap.append(temp_bgr)


    def _find_all_regions_xy_points(self):

        temp_grid = copy.copy(self._grid_with_regions)

        for ind in range(self._no_of_centroids):
            
            temp_region = np.where(np.all(temp_grid == self._colmap[ind], axis=-1))
            self._regions_xy_points.append(temp_region)
            # temp_grid[temp_region[0][:], temp_region[1][:]] = [255,255,255]

            # cv2.imshow('OCCUPANCY GRID WITH REGIONS', temp_grid)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()


    def _find_grid_with_regions(self):

        self._grid_with_regions = copy.copy(self._grid)

        temp_free_points = pd.DataFrame({
        'x': self._free_points[1],
        'y': self._free_points[0]
        })

        kmeans = KMeans(n_clusters=self._no_of_centroids)
        kmeans.fit(temp_free_points)

        labels = kmeans.predict(temp_free_points)

        self._centroids = np.array(kmeans.cluster_centers_, dtype=np.int)

        colors = list(map(lambda x: self._colmap[x], labels))

        for i in range(len(self._free_points[0])):
            self._grid_with_regions[self._free_points[0][i], self._free_points[1][i]] = colors[i]


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

        # cv2.imshow('Occupancy_grid', temp_grid)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


    def _get_random_centroids(self):

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
