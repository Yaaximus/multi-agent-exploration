import copy

import numpy as np

from cv2 import cv2
from utils.graph import Node, Graph
from utils.util_functions import stateNameToCoords

class GridWorld(Graph):
    def __init__(self, x_dim, y_dim, edge_cost, grid):
        self._grid = grid
        self._x_dim = x_dim
        self._y_dim = y_dim
        self._edge_cost = edge_cost
        
        self._cells = [0] * self._y_dim
        
        for i in range(self._y_dim):
            self._cells[i] = [0] * self._x_dim
            
        self.graph = {}
        
        self._generate_graph_from_grid()
        
#         for el in self._cells:
#             print(el)

    def _generate_graph_from_grid(self):
        
        temp_grid = copy.copy(self._grid)
        print(np.shape(temp_grid))
        
        pos_x = self._edge_cost
        pos_y = self._edge_cost
        
        for i in range(self._y_dim):
            for j in range(self._x_dim):
                temp_grid = cv2.circle(temp_grid, (pos_x,pos_y), 1, [0,255,0])
                pos_x += self._edge_cost
            pos_x = self._edge_cost
            pos_y += self._edge_cost
            
        
        for i in range(self._y_dim):
            for j in range(self._x_dim):
                node = Node('x'+str(j)+'y'+str(i))
                current_node_name = 'x'+str(j)+'y'+str(i)
#                 print(current_node_name)
                if j>0: # not top row
                    
                    node_under_tesing_name = 'x'+str(j-1)+'y'+str(i)
#                     print("\t", node_under_tesing_name)
                    if self._check_if_obs_bw_nodes(current_node_name, node_under_tesing_name):
                        node.parents[node_under_tesing_name] = self._edge_cost
                        node.children[node_under_tesing_name] = self._edge_cost
                    else:
                        temp_coord_1 = stateNameToCoords(current_node_name, self._edge_cost)
                        temp_coord_2 = stateNameToCoords(node_under_tesing_name, self._edge_cost)
                        temp_grid = cv2.line(temp_grid, (temp_coord_1[0],temp_coord_1[1]),\
                                     (temp_coord_2[0],temp_coord_2[1]), [255,0,0], 2)
                
                if j+1 < self._y_dim: # not bottom row
                    
                    node_under_tesing_name = 'x'+str(j+1)+'y'+str(i)
#                     print("\t", node_under_tesing_name)
                    if self._check_if_obs_bw_nodes(current_node_name, node_under_tesing_name):
                        node.parents[node_under_tesing_name] = self._edge_cost
                        node.children[node_under_tesing_name] = self._edge_cost
                    else:
                        temp_coord_1 = stateNameToCoords(current_node_name, self._edge_cost)
                        temp_coord_2 = stateNameToCoords(node_under_tesing_name, self._edge_cost)
                        temp_grid = cv2.line(temp_grid, (temp_coord_1[0],temp_coord_1[1]),\
                                     (temp_coord_2[0],temp_coord_2[1]), [255,0,0], 2)
                
                if i>0: # not left most col
                    
                    node_under_tesing_name = 'x'+str(j)+'y'+str(i-1)
#                     print("\t", node_under_tesing_name)
                    if self._check_if_obs_bw_nodes(current_node_name, node_under_tesing_name):
                        node.parents[node_under_tesing_name] = self._edge_cost
                        node.children[node_under_tesing_name] = self._edge_cost
                    else:
                        temp_coord_1 = stateNameToCoords(current_node_name, self._edge_cost)
                        temp_coord_2 = stateNameToCoords(node_under_tesing_name, self._edge_cost)
                        temp_grid = cv2.line(temp_grid, (temp_coord_1[0],temp_coord_1[1]),\
                                             (temp_coord_2[0],temp_coord_2[1]), [255,0,0], 2)
                
                if i+1 < self._x_dim: # not right most
                    
                    node_under_tesing_name = 'x'+str(j)+'y'+str(i+1)
#                     print("\t", node_under_tesing_name)
                    if self._check_if_obs_bw_nodes(current_node_name, node_under_tesing_name):
                        node.parents[node_under_tesing_name] = self._edge_cost
                        node.children[node_under_tesing_name] = self._edge_cost
                    else:
                        temp_coord_1 = stateNameToCoords(current_node_name, self._edge_cost)
                        temp_coord_2 = stateNameToCoords(node_under_tesing_name, self._edge_cost)
                        temp_grid = cv2.line(temp_grid, (temp_coord_1[0],temp_coord_1[1]),\
                                     (temp_coord_2[0],temp_coord_2[1]), [255,0,0], 2)
                
                self.graph['x'+str(j)+'y'+str(i)] = node
                
        cv2.imshow('Occupancy_grid',temp_grid)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
                
    
    def show_graph_on_occupncy_grid(self):
        temp_grid = copy.copy(self._grid)
        
        pos_x = self._edge_cost
        pos_y = self._edge_cost
        
        for i in range(self._y_dim):
            for j in range(self._x_dim):
                temp_grid = cv2.circle(temp_grid, (pos_x,pos_y), 1, [0,255,0])
                pos_x += self._edge_cost
            pos_x = self._edge_cost
            pos_y += self._edge_cost
        
        cv2.imshow('Occupancy_grid',temp_grid)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    
    def _check_if_obs_bw_nodes(self, node_1, node_2):
        
        temp_coord_1 = stateNameToCoords(node_1, self._edge_cost)
        temp_coord_2 = stateNameToCoords(node_2, self._edge_cost)
        
        if temp_coord_1[0] == temp_coord_2[0]:
            width_min = temp_coord_1[0] - 2
            width_max = temp_coord_2[0] + 2
            temp_width_pxls = temp_coord_1[0]*np.ones(shape=[1, self._edge_cost], dtype=np.int)
        else:
            width_min = min(temp_coord_1[0], temp_coord_2[0])
            width_max = max(temp_coord_1[0], temp_coord_2[0])
            temp_width_pxls = np.array(range(min(temp_coord_1[0], temp_coord_2[0]), \
                                             max(temp_coord_1[0], temp_coord_2[0]),1))
        if temp_coord_1[1] == temp_coord_2[1]:
            len_min = temp_coord_1[1] - 2
            len_max = temp_coord_2[1] + 2
            temp_len_pxls = temp_coord_1[1]*np.ones(shape=[1, self._edge_cost], dtype=np.int)
        else:
            len_min = min(temp_coord_1[1], temp_coord_2[1])
            len_max = max(temp_coord_1[1], temp_coord_2[1])
            temp_len_pxls = np.array(range(min(temp_coord_1[1], temp_coord_2[1]), \
                                           max(temp_coord_1[1], temp_coord_2[1]),1))

        temp_grid = copy.copy(self._grid)
        
        self._roi = copy.copy(temp_grid[len_min:len_max, width_min:width_max])
        temp_points = np.where(np.all(self._roi == [0, 0, 0], axis=-1))
        
        if len(temp_points[0])>0 or len(temp_points[1])>0:
#             print("Obstacle")
#             print(temp_coord_1, temp_coord_2)
            return False
#             print(temp_points[0], temp_points[1])
        else:
#             print("Path Clear")
            return True
