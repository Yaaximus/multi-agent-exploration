# Things to optimize
# - Energy cost
# - Time cost
# - Distance cost

import copy

import numpy as np

from cv2 import cv2
from config.Config import Config
from scipy.optimize import linear_sum_assignment


class HungarianRegionAssignment(object):


    def __init__(self, cost_matrix, occupancy_grid, agenthandler, region_centroids):

        self._row_ind = None
        self._col_ind = None
        self._cost_matrix = cost_matrix
        self._agenthandler = agenthandler
        self._grid = copy.copy(occupancy_grid)
        self._region_centroids = region_centroids
        self._grid_with_regions_and_agents = None
        
        self._agent_color_list = self._agenthandler.get_all_agent_color_list()

    
    def find_regions(self):

        self._row_ind, self._col_ind = linear_sum_assignment(self._cost_matrix)


    def get_regions(self):

        return self._row_ind, self._col_ind

    
    def get_total_cost(self):

        return self._cost_matrix[self._row_ind, self._col_ind].sum()

    
    def get_grid_with_regions_and_agents(self):

        return self._grid_with_regions_and_agents


    def place_agents_on_grid_with_info_of_assigned_region(self):

        temp_grid = copy.copy(self._grid)

        font = cv2.FONT_HERSHEY_SIMPLEX
        
        for i in range(Config.NO_OF_AGENTS):
            
            temp_agent_pos = self._agenthandler.get_pos_of_agent(i)
            a = np.array([temp_agent_pos['x'], temp_agent_pos['y']])
            string_to_add = "Agent:" + str(i) + ", region:" + str(self._col_ind[i])
            temp_grid = cv2.putText(temp_grid, string_to_add,(a[0],a[1]+25), font, 0.5,(0,0,0),2,cv2.LINE_AA)

            color_b,color_g,color_r = self._agent_color_list[i]

            temp_grid = cv2.ellipse(temp_grid,(a[0],a[1]),(10,10),0,15,345,(color_b,color_g,color_r),-1)

        for j in range(len(self._region_centroids)):
            b = np.array(self._region_centroids[j])
            string_to_add = "Region" + str(j)
            temp_grid = cv2.putText(temp_grid, string_to_add,(b[0],b[1]-20), font, 0.5,(0,0,0),2,cv2.LINE_AA)

        self._grid_with_regions_and_agents = temp_grid        

    
    def show_assigned_regions(self):

        cv2.imshow('Occupancy_grid with agents and regions', self._grid_with_regions_and_agents)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
