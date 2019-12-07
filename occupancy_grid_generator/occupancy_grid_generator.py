import copy

import numpy as np

from cv2 import cv2
from config.Config import Config


class OccupancyGridGenerator(object):

    def __init__(self):
        
        self._grid_len = Config.GRID_LEN
        self._grid_width = Config.GRID_WIDTH
        self._free_space = Config.FREE_SPACE
        self._complexity_level = Config.COMPLEXITY_LEVEL
        self._grid_without_obs = None
        self._grid_with_obs = None


    def _add_entrance_into_map(self):

        temp_grid = copy.copy(self._grid_without_obs)

        temp_grid = cv2.line(temp_grid, (self._free_space,self._free_space), (self._grid_width-self._free_space,self._free_space), (0,0,0), 4)
        temp_grid = cv2.line(temp_grid, (self._free_space,self._grid_len-self._free_space), (self._grid_width-self._free_space,self._grid_len-self._free_space), (0,0,0), 4)
        temp_grid = cv2.line(temp_grid, (self._free_space,self._free_space), (self._free_space, int(self._grid_len/2)-50), (0,0,0), 4)
        temp_grid = cv2.line(temp_grid, (self._free_space,int(self._grid_len/2)+50), (self._free_space,self._grid_len-self._free_space), (0,0,0), 4)
        temp_grid = cv2.line(temp_grid, (self._grid_width-self._free_space,self._free_space), (self._grid_width-self._free_space, int(self._grid_len/2)-50), (0,0,0), 4)
        temp_grid = cv2.line(temp_grid, (self._grid_width-self._free_space,int(self._grid_len/2)+50), (self._grid_width-self._free_space, self._grid_len-self._free_space), (0,0,0), 4)
        
        self._grid_without_obs = temp_grid

    
    def _generate_easy_grid(self):

        temp_grid = copy.copy(self._grid_without_obs)

        temp_grid = cv2.line(temp_grid, (int(self._grid_width/2),150), (int(self._grid_width/2),self._grid_len-150), (0,0,0), 4)
        temp_grid = cv2.line(temp_grid, (150, int(self._grid_len/2)), (self._grid_width-150,int(self._grid_len/2)), (0,0,0), 4)

        self._grid_without_obs = temp_grid
        self._add_ostacles(no_of_obs=2)

    
    def _generate_moderate_grid(self):

        temp_grid = copy.copy(self._grid_without_obs)

        temp_grid = cv2.line(temp_grid, (int(self._grid_width/2),150), (int(self._grid_width/2),self._grid_len-150), (0,0,0), 4)
        temp_grid = cv2.line(temp_grid, (150, int(self._grid_len/2)), (self._grid_width-150,int(self._grid_len/2)), (0,0,0), 4)
        temp_grid = cv2.line(temp_grid, (int((self._grid_width/2)-self._grid_width/5), self._free_space), (int((self._grid_width/2)-self._grid_width/5), int(self._grid_len/4)), (0,0,0), 4)
        temp_grid = cv2.line(temp_grid, (int((self._grid_width/2)+self._grid_width/5), self._free_space), (int((self._grid_width/2)+self._grid_width/5), int(self._grid_len/4)), (0,0,0), 4)
        temp_grid = cv2.line(temp_grid, (int((self._grid_width/2)-self._grid_width/5), self._grid_len-self._free_space), (int((self._grid_width/2)-self._grid_width/5), int(self._grid_len-(self._grid_len/4))), (0,0,0), 4)
        temp_grid = cv2.line(temp_grid, (int((self._grid_width/2)+self._grid_width/5), self._grid_len-self._free_space), (int((self._grid_width/2)+self._grid_width/5), int(self._grid_len-(self._grid_len/4))), (0,0,0), 4)
        
        self._grid_without_obs = temp_grid
        self._add_ostacles(no_of_obs=3)

    
    def _add_ostacles(self, no_of_obs=1):

        temp_grid = copy.copy(self._grid_without_obs)

        for ind in range(int(no_of_obs)):
            
            random_x = np.random.randint(200, self._grid_width-200)
            random_y = np.random.randint(200, self._grid_len-200)

            pts = np.array([[random_x, random_y],[random_x+30, random_y],[random_x+30, random_y-30],[random_x, random_y-30]], np.int32)
            
            cv2.fillPoly(temp_grid, [pts], 0)

        self._grid_with_obs = temp_grid


    def _generate_difficult_grid(self):
        
        self._generate_moderate_grid()
        self._add_ostacles(no_of_obs=4)

    
    def generate_occupancy_grid(self):

        self._grid_without_obs = 255 * np.ones(shape=[self._grid_len, self._grid_width, 3], dtype=np.uint8)
        # Add boarder of occupancy grid
        self._grid_without_obs = cv2.copyMakeBorder(self._grid_without_obs, 10, 10, 10, 10, cv2.BORDER_CONSTANT)
        self._add_entrance_into_map()

        if self._complexity_level == "very easy":
            self._add_ostacles(no_of_obs=1)

        elif self._complexity_level == "easy":
            self._generate_easy_grid()

        elif self._complexity_level == "moderate":
            self._generate_moderate_grid()

        elif self._complexity_level == "difficult":
            self._generate_difficult_grid()

        else:
            self._add_ostacles(no_of_obs=1)


    def show_occupancy_grid_with_obs(self):

        cv2.imshow('Occupancy_grid',self._grid_with_obs)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    
    def show_occupancy_grid_without_obs(self):

        cv2.imshow('Occupancy_grid',self._grid_without_obs)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    
    def get_occupancy_grid_with_obs(self):

        return self._grid_with_obs


    def get_occupancy_grid_without_obs(self):

        return self._grid_without_obs