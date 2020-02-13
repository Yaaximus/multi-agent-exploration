import copy

import numpy as np

from cv2 import cv2

from config.Config import Config
from utils.util_functions import stateNameToCoords, getRowColumnFromName


class Mapper(object):

    def __init__(self, global_grid, agent_handler):

        self._no_of_agents = Config.NO_OF_AGENTS
        self._grid_len = Config.GRID_LEN
        self._grid_width = Config.GRID_WIDTH
        self._sensor_range = Config.SENSOR_RANGE
        self._agent_color_list = agent_handler.get_all_agent_color_list()
        self._global_grid = copy.copy(global_grid)
        self._mapped_grid = np.zeros(shape=[self._grid_len, self._grid_width, 3], dtype=np.uint8)
        self._mapped_grid = cv2.copyMakeBorder(self._mapped_grid, 10, 10, 10, 10, cv2.BORDER_CONSTANT)

    
    def _check_if_new_obs_on_any_node(self, graph, start_x, end_x, start_y, end_y):

        list_of_nodes_to_check = []

        # print("Graph:", graph.graph)

        for el in graph.graph:
            temp_coord = stateNameToCoords(el, Config.EDGE_COST)
            if temp_coord[1] > min(start_x, end_x) and temp_coord[1] < max(start_x, end_x):
                if temp_coord[0] > min(start_y, end_y) and temp_coord[0] < max(start_y, end_y):
                    list_of_nodes_to_check.append(el)

        temp_grid = copy.copy(self._global_grid)
        
        roi = copy.copy(temp_grid[start_y: end_y, start_x: end_x])
        
        # cv2.imshow('ROI',roi)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        temp_list_of_node_to_remove = []

        temp_points = np.where(np.all(roi == [0, 0, 0], axis=-1))
        
        if len(temp_points[0])>0 or len(temp_points[1])>0:
            # print("Obstacle")
            for el in list_of_nodes_to_check:
                temp_coord = stateNameToCoords(el, Config.EDGE_COST)
                # print("Coord:", temp_coord[0], temp_coord[1])
                # print("Pixels:", temp_grid[temp_coord[0], temp_coord[1]])
                
                temp = np.where(np.all(temp_grid[temp_coord[0], temp_coord[1]] == [0, 0, 0], axis=-1))[0]
                # print("Pixel Ind:", temp)
                if len(temp)>0:
                    # print("List of Nodes to Check:", list_of_nodes_to_check)
                    temp_row, temp_column = getRowColumnFromName(el)
                    # print("row:", temp_row, ", col:", temp_column)
                    # print(graph.cells[temp_row][temp_column])
                    graph.cells[temp_row][temp_column] = -1
                    # print(graph.cells[temp_row][temp_column])
                    # print("Mapper:map_grid:_check_if_new_obs_on_any_node:WARNING: Obstacle detected on cell: row:", temp_row, ", col:", temp_column)
                    # input("Press a key to continue...")
                    # print(graph.cells)
                    # graph.plot_graph_status(graph.graph)
                    temp_list_of_node_to_remove.append(el)
            return graph, temp_list_of_node_to_remove
        else:
            # print("Path Clear")
            return graph, temp_list_of_node_to_remove


    def map_grid(self, agent_no, agent_pos, agent_last_pos, graph):

        # temp_grid = copy.copy(self._mapped_grid)

        start_x = agent_pos['x'] - self._sensor_range
        end_x = agent_pos['x'] + self._sensor_range

        start_y = agent_pos['y'] - self._sensor_range
        end_y = agent_pos['y'] + self._sensor_range
	
        # print("Agent:", agent_no)
        graph, list_of_node_to_remove = self._check_if_new_obs_on_any_node(graph, start_x, end_x, start_y, end_y)

        if start_x < 0:
            start_x = 0

        if end_x > self._grid_width:
            end_x = self._grid_width

        if start_y < 0:
            start_y = 0

        if end_y > self._grid_len:
            end_y = self._grid_len
        self._global_grid = cv2.circle(self._global_grid,(agent_pos['x'], agent_pos['y']),2,(self._agent_color_list[agent_no]))

        try:
            # print(agent_pos['x'], agent_pos['y'], agent_last_pos['x'], agent_last_pos['y'])
            self._global_grid = cv2.line(self._global_grid, (agent_pos['x'], agent_pos['y']), \
                (agent_last_pos['x'], agent_last_pos['y']), (self._agent_color_list[agent_no]), 2)
        except TypeError:
            # print(agent_pos['x'], agent_pos['y'], agent_last_pos[1], agent_last_pos[0])
            self._global_grid = cv2.line(self._global_grid, (agent_pos['x'], agent_pos['y']), \
                (agent_last_pos[1], agent_last_pos[0]), (self._agent_color_list[agent_no]), 2)

        self._mapped_grid[start_y: end_y, start_x: end_x] = self._global_grid[start_y: end_y, start_x: end_x]
        # cv2.ellipse(self._mapped_grid,(agent_pos['x'], agent_pos['y']),(10,10),0,15,345,(self._agent_color_list[agent_no]),-1)

        return graph, list_of_node_to_remove
    
    
    def show_mapped_grid(self):

        cv2.imshow('Occupancy_grid', self._mapped_grid)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    
    def get_mapped_grid(self):

        return self._mapped_grid
