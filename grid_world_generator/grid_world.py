import copy

import numpy as np
from cv2 import cv2
from config.Config import Config
from utils.graph import Node, Graph
from utils.util_functions import stateNameToCoords


class GridWorld(Graph):
    
    def __init__(self, x_dim, y_dim, grid):
        
        self.graph = {}
        self._grid = grid
        self._x_dim = x_dim
        self._y_dim = y_dim
        self._grid_with_nodes = None
        self.cells = [0] * self._y_dim
        self._edge_cost = Config.EDGE_COST
        self._grid_with_nodes_and_edges = None
        self._grid_with_nodes_and_edges_with_obs = None

        for i in range(self._y_dim):
            self.cells[i] = [0] * self._x_dim


    def plot_graph_status(self, graph):
        
        self._draw_all_traversable_edges_on_grid(graph)
        self.show_nodes_and_all_traversable_edges()
        
    
    def _draw_all_traversable_edges_on_grid(self, graph=None):
        
        temp_grid = copy.copy(self._grid_with_nodes)
        if graph is None:
            temp_graph = self.graph
        else:
            temp_graph = graph
	
        
        for el in temp_graph:
            # print(el, temp_graph[str(el)].children)
            for child in temp_graph[str(el)].children:
                temp_coord_1 = stateNameToCoords(temp_graph[str(el)].id, self._edge_cost)
                temp_coord_2 = stateNameToCoords(child, self._edge_cost)
                # print("\t", el, child, (temp_coord_1[1],temp_coord_1[0]), \
                #      (temp_coord_2[1],temp_coord_2[0]))
                temp_grid = cv2.line(temp_grid, (temp_coord_1[1],temp_coord_1[0]),\
                         (temp_coord_2[1],temp_coord_2[0]), [255,0,0], 2)
                
        self._grid_with_nodes_and_all_traversable_edges = temp_grid

        
    def _generate_graph(self):
        
        temp_grid = copy.copy(self._grid_with_nodes)

        for i in range(len(self.cells)):
            row = self.cells[i]
            for j in range(len(row)):
                
                node = Node('x'+str(i)+'y'+str(j))
                current_node_name = 'x'+str(i)+'y'+str(j)
                if i>0: # not top row
                    node_under_tesing_name = 'x'+str(i-1)+'y'+str(j)
                    node, temp_grid = self._decide_if_connection_or_not(\
                                            current_node_name, node_under_tesing_name, node, temp_grid)
                if i+1 < self._y_dim: # not bottom row
                    node_under_tesing_name = 'x'+str(i+1)+'y'+str(j)
                    node, temp_grid = self._decide_if_connection_or_not(\
                                            current_node_name, node_under_tesing_name, node, temp_grid)
                if j>0: # not left most col
                    node_under_tesing_name = 'x'+str(i)+'y'+str(j-1)
                    node, temp_grid = self._decide_if_connection_or_not(\
                                            current_node_name, node_under_tesing_name, node, temp_grid)
                if j+1 < self._x_dim: # not right most
                    node_under_tesing_name = 'x'+str(i)+'y'+str(j+1)
                    node, temp_grid = self._decide_if_connection_or_not(\
                                            current_node_name, node_under_tesing_name, node, temp_grid)
                
                self.graph['x'+str(i)+'y'+str(j)] = node
                
        self._grid_with_nodes_and_edges_with_obs = temp_grid
    
    
    def _decide_if_connection_or_not(self, current_node_name, node_under_tesing_name, node, temp_grid):
                        
        if self._check_if_no_obs_bw_nodes(current_node_name, node_under_tesing_name):
            node.parents[node_under_tesing_name] = self._edge_cost
            node.children[node_under_tesing_name] = self._edge_cost
        else:
            temp_coord_1 = stateNameToCoords(current_node_name, self._edge_cost)
            temp_coord_2 = stateNameToCoords(node_under_tesing_name, self._edge_cost)
            # print("temp_coord_1:", temp_coord_1, ", temp_coord_2:", temp_coord_2)
            temp_grid = cv2.line(temp_grid, (temp_coord_1[1],temp_coord_1[0]),\
                         (temp_coord_2[1],temp_coord_2[0]), [0,0,255], 2)

        return node, temp_grid
                
    
    def _draw_nodes_on_grid(self):
        
        temp_grid = copy.copy(self._grid)
        
        pos_x = self._edge_cost
        pos_y = self._edge_cost
        
        for _ in range(self._y_dim):
            for _ in range(self._x_dim):
                temp_grid = cv2.circle(temp_grid, (pos_x,pos_y), 3, [0,210,0])
                pos_x += self._edge_cost
            pos_x = self._edge_cost
            pos_y += self._edge_cost
            
        self._grid_with_nodes = temp_grid
        
    
    def _check_if_no_obs_bw_nodes(self, node_1, node_2):
        
        # print(node_1, node_2)
        
        temp_coord_1 = stateNameToCoords(node_1, self._edge_cost)
        temp_coord_2 = stateNameToCoords(node_2, self._edge_cost)
        
        if temp_coord_1[1] == temp_coord_2[1]:
            width_min = temp_coord_1[1] - 5
            width_max = temp_coord_2[1] + 5
        else:
            width_min = min(temp_coord_1[1], temp_coord_2[1]) - 5
            width_max = max(temp_coord_1[1], temp_coord_2[1]) + 5
        if temp_coord_1[0] == temp_coord_2[0]:
            len_min = temp_coord_1[0] - 5
            len_max = temp_coord_2[0] + 5
        else:
            len_min = min(temp_coord_1[0], temp_coord_2[0]) - 5
            len_max = max(temp_coord_1[0], temp_coord_2[0]) + 5

        temp_grid = copy.copy(self._grid)
        
        self._roi = copy.copy(temp_grid[len_min:len_max, width_min:width_max])
        temp_points = np.where(np.all(self._roi == [0, 0, 0], axis=-1))
        
        if len(temp_points[0])>0 or len(temp_points[1])>0:# print("Obstacle")
            return False
        else:                                             # print("Path Clear")
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
        
    
    def get_occupancy_grid_with_nodes(self):
        
        return self._grid_with_nodes
    
    
    def get_occupancy_grid_with_nodes_and_edges_with_obs(self):
        
        return self._grid_with_nodes_and_edges_with_obs
        
    
    def get_occupancy_grid_with_nodes_and_all_traversable_edges(self):
        
        return self._grid_with_nodes_and_all_traversable_edges

    
    def get_graph(self):

        return self.graph
        
    
    def run(self):
        
        self._draw_nodes_on_grid()
        self._generate_graph()
        self._draw_all_traversable_edges_on_grid()
