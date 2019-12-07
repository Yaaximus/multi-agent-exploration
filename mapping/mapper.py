import copy

import numpy as np

from cv2 import cv2

from config.Config import Config


class Mapper(object):

    def __init__(self, global_grid):

        self._no_of_agents = Config.NO_OF_AGENTS
        self._grid_len = Config.GRID_LEN
        self._grid_width = Config.GRID_WIDTH
        self._sensor_range = Config.SENSOR_RANGE
        self._global_grid = copy.copy(global_grid)
        self._mapped_grid = np.zeros(shape=[self._grid_len, self._grid_width, 3], dtype=np.uint8)
        self._mapped_grid = cv2.copyMakeBorder(self._mapped_grid, 10, 10, 10, 10, cv2.BORDER_CONSTANT)


    def map_grid(self, agent_no, agent_pos):

        start_x = agent_pos['x'] - self._sensor_range
        end_x = agent_pos['x'] + self._sensor_range

        start_y = agent_pos['y'] - self._sensor_range
        end_y = agent_pos['y'] + self._sensor_range

        if start_x < 0:
            start_x = 0

        if end_x > self._grid_width:
            end_x = self._grid_width

        if start_y < 0:
            start_y = 0

        if end_y > self._grid_len:
            end_y = self._grid_len

        if agent_no == 0:
            color_b,color_g,color_r = 255,0,0
        elif agent_no == 1:
            color_b,color_g,color_r = 0,255,0
        else:
            color_b,color_g,color_r = 0,0,255

        self._mapped_grid[start_y: end_y, start_x: end_x] = self._global_grid[start_y: end_y, start_x: end_x]
        cv2.ellipse(self._mapped_grid,(agent_pos['x'], agent_pos['y']),(10,10),0,15,345,(color_b,color_g,color_r),-1)
    
    
    def show_mapped_grid(self):

        cv2.imshow('Occupancy_grid',self._mapped_grid)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    
    def get_mapped_grid(self):

        return self._mapped_grid