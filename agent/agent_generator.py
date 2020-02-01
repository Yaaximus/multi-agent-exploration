import copy

import numpy as np

from cv2 import cv2
from config.Config import Config

from utils.util_functions import l2_distance
from utils.graph import get_closest_vertex_coords_on_graph_from_pos

class AgentGenerator(object):

    def __init__(self, graph):

        self._grid_len = Config.GRID_LEN
        self._grid_width = Config.GRID_WIDTH
        self._free_space = Config.FREE_SPACE
        self._edge_cost = Config.EDGE_COST
        self._agent_pos = {'x':None, 'y':None}
        self._graph = graph

    
    def generate_agent(self):

        random_x = np.random.randint(20, self._grid_width-20)

        if random_x < (self._free_space -20) or random_x > (self._grid_width - self._free_space + 20):
            random_y = np.random.randint(20, self._grid_len-20)
        else:
            temp = np.random.randint(0,2)
            if temp == 0:
                random_y = np.random.randint(20, self._free_space-20)
            else:
                random_y = np.random.randint(self._grid_len - self._free_space + 20, self._grid_len - 20)

        self._agent_pos['x'], self._agent_pos['y'] = get_closest_vertex_coords_on_graph_from_pos(self._graph, random_x, random_y, self._edge_cost)


    def get_agent_pos(self):

        return self._agent_pos

    
    def move_agent(self, x, y):

        temp_x = self._agent_pos['x']
        temp_y = self._agent_pos['y']

        if l2_distance(temp_x, temp_y, temp_x+x, temp_y+y)<10:
            self._agent_pos['x'] = self._agent_pos['x'] + x
            self._agent_pos['y'] = self._agent_pos['y'] + y
        else:
            
            if x > 0:
                x_to_add = 10
            else:
                x_to_add = -10

            if y > 0:
                y_to_add = 10
            else:
                y_to_add = -10

            self._agent_pos['x'] = self._agent_pos['x'] + x_to_add
            self._agent_pos['y'] = self._agent_pos['y'] + y_to_add

    
    def set_pos_of_agent(self, new_x, new_y):

        self._agent_pos['x'] = new_x
        self._agent_pos['y'] = new_y
