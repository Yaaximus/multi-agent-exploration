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


    def __init__(self, cost_matrix, occupancy_grid):

        self._row_ind = None
        self._col_ind = None
        self._grid = copy.copy(occupancy_grid)
        self._cost_matrix = cost_matrix

    
    def find_regions(self):

        self._row_ind, self._col_ind = linear_sum_assignment(self._cost_matrix)


    def get_regions(self):

        return self._row_ind, self._col_ind

    
    def get_total_cost(self):

        return self._cost_matrix[self._row_ind, self._col_ind].sum()

    
    def show_assigned_regions(self, agenthandler, region_centroids):

        temp_grid = copy.copy(self._grid)
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        for i in range(Config.NO_OF_AGENTS):
            
            temp_agent_pos = agenthandler.get_pos_of_agent(i)
            a = np.array([temp_agent_pos['x'], temp_agent_pos['y']])
            string_to_add = "Agent:" + str(i) + ", region:" + str(self._col_ind[i])
            temp_grid = cv2.putText(temp_grid, string_to_add,(a[0],a[1]+25), font, 0.5,(0,0,0),2,cv2.LINE_AA)

            if i == 0:
                color_b,color_g,color_r = 255,0,0
            elif i == 1:
                color_b,color_g,color_r = 0,255,0
            else:
                color_b,color_g,color_r = 0,0,255

            temp_grid = cv2.ellipse(temp_grid,(a[0],a[1]),(10,10),0,15,345,(color_b,color_g,color_r),-1)

        for j in range(len(region_centroids)):
            b = np.array(region_centroids[j])
            string_to_add = "Region" + str(j)
            temp_grid = cv2.putText(temp_grid, string_to_add,(b[0],b[1]-20), font, 0.5,(0,0,0),2,cv2.LINE_AA)

        cv2.imshow('Occupancy_grid', temp_grid)
        cv2.waitKey(0)
        cv2.destroyAllWindows()