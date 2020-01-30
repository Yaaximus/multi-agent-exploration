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
        self._grid_with_nodes = None
        self._grid_with_nodes_and_edges = None
        self._grid_with_nodes_and_edges_with_obs = None
        
    
    def _draw_all_traversable_edges_on_grid(self):
        
        temp_grid = copy.copy(self._grid_with_nodes)
        
        for el in self.graph:
            for child in self.graph[str(el)].children:
                temp_coord_1 = stateNameToCoords(self.graph[str(el)].id, self._edge_cost)
                temp_coord_2 = stateNameToCoords(child, self._edge_cost)
                temp_grid = cv2.line(temp_grid, (temp_coord_1[0],temp_coord_1[1]),\
                         (temp_coord_2[0],temp_coord_2[1]), [255,0,0], 2)
                
        self._grid_with_nodes_and_all_traversable_edges = temp_grid

        
    def _generate_graph(self):
        
        temp_grid = copy.copy(self._grid_with_nodes)
        
        for i in range(self._y_dim+1):
            for j in range(self._x_dim+1):
                node = Node('x'+str(j)+'y'+str(i))
                current_node_name = 'x'+str(j)+'y'+str(i)
                if j>0: # not top row
                    node_under_tesing_name = 'x'+str(j-1)+'y'+str(i)
                    node, temp_grid = self._decide_if_connection_or_not(\
                                            current_node_name, node_under_tesing_name, node, temp_grid)
                if j+1 < self._y_dim: # not bottom row
                    node_under_tesing_name = 'x'+str(j+1)+'y'+str(i)
                    node, temp_grid = self._decide_if_connection_or_not(\
                                            current_node_name, node_under_tesing_name, node, temp_grid)
                if i>0: # not left most col
                    node_under_tesing_name = 'x'+str(j)+'y'+str(i-1)
                    node, temp_grid = self._decide_if_connection_or_not(\
                                            current_node_name, node_under_tesing_name, node, temp_grid)
                if i+1 < self._x_dim: # not right most
                    node_under_tesing_name = 'x'+str(j)+'y'+str(i+1)
                    node, temp_grid = self._decide_if_connection_or_not(\
                                            current_node_name, node_under_tesing_name, node, temp_grid)
                
                self.graph['x'+str(j)+'y'+str(i)] = node
                
        self._grid_with_nodes_and_edges_with_obs = temp_grid
    
    
    def _decide_if_connection_or_not(self, current_node_name, node_under_tesing_name, node, temp_grid):
                        
        if self._check_if_no_obs_bw_nodes(current_node_name, node_under_tesing_name):
            node.parents[node_under_tesing_name] = self._edge_cost
            node.children[node_under_tesing_name] = self._edge_cost
        else:
            temp_coord_1 = stateNameToCoords(current_node_name, self._edge_cost)
            temp_coord_2 = stateNameToCoords(node_under_tesing_name, self._edge_cost)
            temp_grid = cv2.line(temp_grid, (temp_coord_1[0],temp_coord_1[1]),\
                         (temp_coord_2[0],temp_coord_2[1]), [255,0,0], 2)

        return node, temp_grid
                
    
    def _draw_nodes_on_grid(self):
        
        temp_grid = copy.copy(self._grid)
        
        pos_x = self._edge_cost
        pos_y = self._edge_cost
        
        for i in range(self._y_dim):
            for j in range(self._x_dim):
                temp_grid = cv2.circle(temp_grid, (pos_x,pos_y), 3, [0,210,0])
                pos_x += self._edge_cost
            pos_x = self._edge_cost
            pos_y += self._edge_cost
            
        self._grid_with_nodes = temp_grid
        
    
    def _check_if_no_obs_bw_nodes(self, node_1, node_2):
        
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
            return False
        else:
#             print("Path Clear")
            return True

    
    def show_nodes_on_occupancy_grid(self):
        
        cv2.imshow('Occupancy_grid_with_nodes', self._grid_with_nodes)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        
    def show_nodes_and_edges_with_obs_on_occupancy_grid(self):
        
        cv2.imshow('Occupancy_grid_with_nodes_and_edges_with_obs', self._grid_with_nodes_and_edges_with_obs)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        
    def show_nodes_and_all_traversable_edges(self):
        
        cv2.imshow('Occupancy_grid_with_nodes_and_all_traversable_edges', self._grid_with_nodes_and_all_traversable_edges)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    
    def run(self):
        
        self._draw_nodes_on_grid()
        self._generate_graph()
        self._draw_all_traversable_edges_on_grid()
