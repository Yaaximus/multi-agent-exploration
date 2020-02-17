import os
import time
import math
import copy

import numpy as np

from cv2 import cv2

from config.Config import Config
from mapping.mapper import Mapper
from utils.graph import check_if_no_obs_bw_nodes
from grid_world_generator.grid_world import GridWorld
from utils.util_functions import stateCoordsToName, stateNameToCoords, l2_distance
from d_star_lite.d_star_lite import initDStarLite, moveAndRescan, runTimeRescanAndMove


class Explorer(object):
    
    def __init__(self, global_grid, known_grid, assigned_region_node_names, graph_list, \
                 agenthandler, color_map, regions_cols, grid_with_regions, grid_with_regions_info): 
        
        self._mapped_grid = None
        self._color_map = color_map
        self._iteration_step = 0
        self._img_with_agents = None
        self._graph_list = graph_list
        self._verbose = Config.VERBOSE
        self._global_grid = global_grid
        self._region_col = regions_cols
        self._grid_len = Config.GRID_LEN
        self._agent_handler = agenthandler
        self._edge_cost = Config.EDGE_COST
        self._grid_width = Config.GRID_WIDTH
        self._font = cv2.FONT_HERSHEY_SIMPLEX
        self._grid_with_regions_display = None
        self._known_grid = copy.copy(known_grid)
        self._no_of_agents = Config.NO_OF_AGENTS
        self._sensor_range = Config.SENSOR_RANGE
        self._k_m_list = [0] * self._no_of_agents
        self._grid_with_regions = grid_with_regions
        self._last_pos = [[]] * self._no_of_agents
        self._queue_list = [[]] * self._no_of_agents
        self._s_new_names = [''] * self._no_of_agents
        self._s_last_names = [''] * self._no_of_agents
        self._s_start_names = [''] * self._no_of_agents
        self._complexity_level = Config.COMPLEXITY_LEVEL
        self._avoiding_obs = [False] * self._no_of_agents
        self._s_current_names = [''] * self._no_of_agents
        self._nodes_to_explore = [[]] * self._no_of_agents
        self._known_grid_display = copy.copy(known_grid)
        self._global_grid_display = copy.copy(global_grid)
        self._grid_with_regions_info = grid_with_regions_info
        self._path_to_save_results = Config.PATH_TO_SAVE_RESULTS
        self._assigned_region_node_names = assigned_region_node_names
        self._agents_path_flow = [[] for _ in range(self._no_of_agents)]
        self._agent_color_list = self._agent_handler.get_all_agent_color_list()
        self._grid_mapper = Mapper(global_grid=self._global_grid, agent_handler=self._agent_handler)
        
        temp_coord = [0,0]
        self._mission_stages_list = []
        for i in range(self._no_of_agents):
            self._mission_stages_list.append({'region_reached':False, 'region_explored':False})
            
        for i in range(self._no_of_agents):
            temp_coord = self._agent_handler.get_pos_of_agent(i)
            self._last_pos[i] = temp_coord
            self._s_start_names[i] = stateCoordsToName(temp_coord['y'], temp_coord['x'], self._edge_cost)
            
        self._s_current_names = self._s_start_names
        for i in range(len(self._graph_list)):
            self._graph_list[i].setStart(self._s_start_names[i])
            self._graph_list[i].setGoal(self._assigned_region_node_names[i])
            
            self._graph_list[i], self._queue_list[i], self._k_m_list[i] = initDStarLite(\
                                                                            self._graph_list[i], \
                                                                            self._queue_list[i], \
                                                                            self._s_start_names[i], \
                                                                            self._assigned_region_node_names[i], \
                                                                            self._k_m_list[i])

        self._path_to_save_simulation_results = os.path.join(self._path_to_save_results, "simulation")
        self._path_to_save_agents_path = os.path.join(self._path_to_save_results, "agents_paths")
        os.makedirs(self._path_to_save_simulation_results, exist_ok=True)
        os.makedirs(self._path_to_save_agents_path, exist_ok=True)

        if self._verbose:
            print("Explorer: START NODE NAMES:\n")
            print(self._s_start_names, "\n")
            print("Explorer: ASSIGNED REGION NODE NAMES:\n")
            print(self._assigned_region_node_names, "\n")
            print("Explorer: Number of Graphs in list:", len(self._graph_list), "\n")

            for i in range(len(self._s_current_names)):
                pos_coords = stateNameToCoords(self._s_current_names[i], self._edge_cost)

                print("Explorer: Node Name: {} , Node Position: {}.".format(self._s_current_names[i], pos_coords))

            print("")
            
    
    def _update_display(self):

        self._iteration_step += 1
        
        self._known_grid_display = copy.copy(self._known_grid)
        self._global_grid_display = copy.copy(self._global_grid)
        self._grid_with_regions_display = copy.copy(self._grid_with_regions_info)
        
        for i in range(self._no_of_agents):
            
            temp_pos = self._agent_handler.get_pos_of_agent(i)
            self._agents_path_flow[i].append(stateCoordsToName(temp_pos['x'], temp_pos['y'], self._edge_cost))
            temp_pos_x = temp_pos['x']
            temp_pos_y = temp_pos['y']
            
            self._update_agents_on_map(i, temp_pos_x, temp_pos_y)
            self._map_world(i, temp_pos)
        
        temp_x, temp_y = 580, 340

        temp_gloabal_grid_with_agents = cv2.resize(self._gloabal_grid_with_agents, (temp_x, temp_y))
        temp_known_grid_with_agents = cv2.resize(self._known_grid_with_agents, (temp_x, temp_y))
        temp_grid_with_agents_and_regions = cv2.resize(self._grid_with_agents_and_regions, (temp_x, temp_y))

        self._mapped_grid = self._grid_mapper.get_mapped_grid()
        temp_mapped_img = cv2.resize(self._mapped_grid, (temp_x, temp_y))

        h1_stack_image = np.hstack((temp_gloabal_grid_with_agents, temp_mapped_img))
        h2_stack_image = np.hstack((temp_known_grid_with_agents, temp_grid_with_agents_and_regions))
        simulation_img = np.vstack((h1_stack_image, h2_stack_image))
        
        temp_file_name = str(self._iteration_step) + ".png"
        cv2.imwrite(os.path.join(self._path_to_save_simulation_results, temp_file_name), simulation_img)

        return simulation_img

    
    def _update_agents_on_map(self, i, temp_pos_x, temp_pos_y):
        
        color_b,color_g,color_r = self._agent_color_list[i]

        self._gloabal_grid_with_agents = cv2.ellipse(self._global_grid_display,(temp_pos_x,temp_pos_y),\
                                            (10,10),0,15,345,(color_b,color_g,color_r),-1)

        self._known_grid_with_agents = cv2.ellipse(self._known_grid_display,(temp_pos_x,temp_pos_y),\
                                            (10,10),0,15,345,(color_b,color_g,color_r),-1)

        string_to_add = "Agent:" + str(i) + ", region:" + str(self._region_col[i])
        self._grid_with_regions_display = cv2.putText(self._grid_with_regions_display, string_to_add,(temp_pos_x,temp_pos_y+25), self._font, 0.5,(0,0,0),2,cv2.LINE_AA)

        self._grid_with_agents_and_regions = cv2.ellipse(self._grid_with_regions_display,(temp_pos_x,temp_pos_y),(10,10),0,15,345,(color_b,color_g,color_r),-1)        
        
    
    def _map_world(self, i, temp_pos):
        
        self._graph_list[i], list_of_node_to_remove = self._grid_mapper.map_grid(agent_no=i, agent_pos=temp_pos, \
                                                                                agent_last_pos=self._last_pos[i], graph=self._graph_list[i])
        
        if len(list_of_node_to_remove)>0:
            for el in list_of_node_to_remove:
                if el in self._nodes_to_explore[i]:
                    self._nodes_to_explore[i].remove(str(el))
        
    
    def _get_nodes_to_explore(self):
        
        for i in range(self._no_of_agents):
            
            temp_list = []
            if self._verbose:
                print("Explorer:_get_nodes_to_explore: Agent {} Color Map:{}".format(i, self._color_map[i]))
            for el in self._graph_list[i].graph:
                temp_coords = stateNameToCoords(el, self._edge_cost)

                width_min = temp_coords[1] - 5
                width_max = temp_coords[1] + 5
                len_min = temp_coords[0] - 5
                len_max = temp_coords[0] + 5

                temp_grid = copy.copy(self._global_grid)
                
                roi = copy.copy(temp_grid[len_min:len_max, width_min:width_max])
                temp_points = np.where(np.all(roi == [0, 0, 0], axis=-1))
                
                if len(temp_points[0])>0 or len(temp_points[1])>0:# print("Obstacle")
                    pass
                else:                                             # print("Path Clear")
                    if np.all(self._grid_with_regions[temp_coords[0], temp_coords[1]] == self._color_map[i], axis=-1):
                        temp_list.append(el)
                    # print(i, el, self._grid_with_regions[temp_coords[0], temp_coords[1]], self._color_map[i])

            self._nodes_to_explore[i] = copy.copy(temp_list)
            # print("Nodes to explore by Agent: {} are {}.".format(i, self._nodes_to_explore[i]))
        if self._verbose: print("")
            
            
    def _get_closest_traversable_node(self, i):
        
        temp_dist = math.inf
        temp_current_coords = stateNameToCoords(self._s_current_names[i], self._edge_cost)
        nodes_to_remove = []
        # print("S Current:", self._s_current_names[i], ", Current Node Coord:", temp_current_coords)
        for el in self._nodes_to_explore[i]:
            temp_coords = stateNameToCoords(el, self._edge_cost)
            new_temp_dist = l2_distance(temp_current_coords[0], temp_current_coords[1], temp_coords[0], temp_coords[1])
    
            if np.all(self._global_grid[temp_coords[0], temp_coords[1]] == [255, 255, 255], axis=-1):
                if new_temp_dist < temp_dist:
                    x = temp_coords[0]
                    y = temp_coords[1]
                    temp_dist = new_temp_dist
            else:
                nodes_to_remove.append(el)
                
        for el in nodes_to_remove:
            self._nodes_to_explore[i].remove(str(el))
        # print("Closest traversable node x:{}, y:{}, ".format(x, y))
        # print("Closest traversable Node Name:{}".format(stateCoordsToName(x, y, self._edge_cost)))
        return stateCoordsToName(x, y, self._edge_cost)


    def _explore_assigned_region(self, i):
        
        self._s_new_names[i] = self._get_closest_traversable_node(i)
        # print("Explorer:_explore_assigned_region: Agent:", i, ", S_current:", self._s_current_names[i], ", S_new", self._s_new_names[i])
        if check_if_no_obs_bw_nodes(self._s_current_names[i], self._s_new_names[i], self._global_grid):
            return True
        else:
            self._avoiding_obs[i] = True
            # print("Explorer:_explore_assigned_region: Obstalce b/w node.")
            X_DIM = int(self._grid_width/self._edge_cost)
            Y_DIM = int(self._grid_len/self._edge_cost)
            temp_graph = GridWorld(X_DIM, Y_DIM, self._known_grid)
            temp_graph.run()
            temp_queue = []
            temp_k_m = 0

            temp_graph.setStart(self._s_current_names[i])
            temp_graph.setGoal(self._s_new_names[i])
            # print("Explorer:_explore_assigned_region: Current Node:", self._s_current_names[i])
            # print("Explorer:_explore_assigned_region: New Goal Node Name:", temp_graph.goal)
            
            temp_graph, temp_queue, temp_k_m = initDStarLite(temp_graph, temp_queue, \
                                                                self._s_current_names[i], \
                                                                temp_graph.goal, \
                                                                temp_k_m)            
            
            return temp_graph, temp_queue, temp_k_m
        
    
    def _check_max_movement_constraint(self, i):
        
        temp_current_coords = stateNameToCoords(self._s_current_names[i], self._edge_cost)
        temp_new_coords = stateNameToCoords(self._s_new_names[i], self._edge_cost)
        temp_dist = l2_distance(temp_current_coords[0], temp_current_coords[1], \
                                    temp_new_coords[0], temp_new_coords[1])
        
        if temp_dist > self._edge_cost:
            temp_dist = math.inf
            for neighbor in self._graph_list[i].graph[self._s_current_names[i]].children:
                # print("Neighbor Name:", neighbor, "Distance from ")
                neighbor_coords = stateNameToCoords(neighbor, self._edge_cost)
                new_temp_dist = l2_distance(temp_new_coords[0], temp_new_coords[1], neighbor_coords[0], neighbor_coords[1])

                if new_temp_dist < temp_dist:
                    if check_if_no_obs_bw_nodes(self._s_current_names[i], neighbor, self._global_grid):
                        x = neighbor_coords[0]
                        y = neighbor_coords[1]
                        temp_dist = new_temp_dist

            self._s_new_names[i] = stateCoordsToName(x, y, self._edge_cost)
            
    
    def _reach_region_and_explore(self):
        
        # TODO: Incase obs on region assigned road handle this situation
        status = [[]] * self._no_of_agents
        self._mission_complete = False
        while not self._mission_complete:
            simulation_img = self._update_display()
            cv2.imshow('MULTI AGENT EXPLORER SIMULATOR', simulation_img)
            k = cv2.waitKey(1) & 0xFF

            if k == ord('q'):
                if self._verbose:
                    print("Mission Aborted!")
                break

            # time.sleep(0.5)
            time.sleep(0.2)

            count = 0
            for el in self._mission_stages_list:
                if el['region_reached'] and el['region_explored']:
                    count += 1
                if count == self._no_of_agents:
                    self._mission_complete = True
                    cv2.imwrite(os.path.join(self._path_to_save_results, "Mapped_Grid.png"), self._mapped_grid)
                    temp_file_name = "Grid_with_regions_explore_with_trajectories.png"
                    cv2.imwrite(os.path.join(self._path_to_save_results, temp_file_name), simulation_img)
                    if self._verbose:
                        print("\nAll agents have Explored Assigned Regions.", "\n")
                        print("Agent's Current Node Names: ", self._s_new_names, "\n")
                        print("-----------------------------------------------------------")
                        print("--------------------Mission Successful---------------------")
                        print("-----------------------------------------------------------")
                    break

            for i in range(len(self._graph_list)):
                if not self._mission_stages_list[i]['region_reached']:

                    self._s_new_names[i], self._k_m_list[i] = moveAndRescan(self._graph_list[i], \
                        self._queue_list[i], self._s_current_names[i], self._sensor_range, self._k_m_list[i],i)

                    temp_coords = stateNameToCoords(self._assigned_region_node_names[i], self._edge_cost)    
                    if np.all(self._mapped_grid[temp_coords[0], temp_coords[1]] == [0, 0, 0], axis=-1):
                        if not self._mission_stages_list[i]['region_reached']:
                            self._mission_stages_list[i]['region_reached'] = True
                            if self._verbose:
                                print("Agent: {} has reached assigned region.".format(i))
                        
                
                if self._mission_stages_list[i]['region_reached'] and not self._mission_stages_list[i]['region_explored']:
                    
                    if not self._avoiding_obs[i]:
                        status[i] = self._explore_assigned_region(i)
                    
                    if status and not self._avoiding_obs[i]:
                        pass
                    else:
                        self._graph_list[i], self._queue_list[i], self._k_m_list[i] = status[i]
                
                        self._s_new_names[i], self._k_m_list[i] = runTimeRescanAndMove(self._graph_list[i], \
                            self._queue_list[i], self._s_current_names[i], self._sensor_range, self._k_m_list[i],i)

                        if self._s_new_names[i] in self._nodes_to_explore[i] or self._s_new_names[i] == 'goal' or self._s_new_names[i] == None:
                            # print("Explorer:_reach_region_and_explore: Exiting...")
                            self._avoiding_obs[i] = False
                            
                if self._s_new_names[i] != 'goal' and self._s_new_names[i] != None:
                    
                    temp_new_coords = stateNameToCoords(self._s_new_names[i], self._edge_cost)
                    if np.all(self._global_grid[temp_new_coords[0], temp_new_coords[1]] == [255, 255, 255], axis=-1):
                        
                        self._check_max_movement_constraint(i)
                        self._last_pos[i] = stateNameToCoords(self._s_current_names[i], self._edge_cost)
                        self._s_current_names[i] = self._s_new_names[i]
                        # print("Explorer:_reach_region_and_explore: s current updated", i, self._s_current_names[i])
                        temp_coord = stateNameToCoords(self._s_current_names[i], self._edge_cost)
                        self._agent_handler.set_pos_of_agent(i, temp_coord[1], temp_coord[0])
                    
                if self._s_current_names[i] in self._nodes_to_explore[i]:
                    self._nodes_to_explore[i].remove(str(self._s_current_names[i]))

                if self._s_current_names[i] == self._assigned_region_node_names[i]:
                    if not self._mission_stages_list[i]['region_reached']:
                        self._mission_stages_list[i]['region_reached'] = True
                        if self._verbose:
                            print("Agent: {} has reached assigned region.".format(i))
                    
                if len(self._nodes_to_explore[i]) == 0:
                    if not self._mission_stages_list[i]['region_explored']:
                        self._mission_stages_list[i]['region_explored'] = True
                        if self._verbose:
                            print("Agent: {} has explored assigned region.".format(i))
                        
                temp_coords = stateNameToCoords(self._s_current_names[i], self._edge_cost)
                if np.all(self._global_grid[temp_coords[0], temp_coords[1]] == [0, 0, 0], axis=-1):
                    if self._verbose: print("On An Obstacle :-P Not Possible.........")
                    while(1):
                        cv2.imshow('MULTI AGENT EXPLORER SIMULATOR', simulation_img)
                        k = cv2.waitKey(1) & 0xFF

                        if k == ord('q'):
                            break

                        time.sleep(0.5)

        # while(1):
        #     if k == ord('q'):
        #         break
        #     else:
        #         cv2.imshow('MULTI AGENT EXPLORER SIMULATOR', simulation_img)
        #         k = cv2.waitKey(1) & 0xFF
        #         time.sleep(1)

        if self._verbose: print("Shutting Down...", "\n")
        cv2.destroyAllWindows()
        if self._verbose: print("Shutting Down Successful.\n")


    def _draw_path_of_all_agents_on_separate_grid(self):

        ind = 0

        for el in self._agents_path_flow:

            temp_grid = copy.copy(self._global_grid)

            for i in range(len(el)-1):
                temp_coord_1 = stateNameToCoords(el[i], self._edge_cost)
                temp_coord_2 = stateNameToCoords(el[i+1], self._edge_cost)

                temp_grid = cv2.line(temp_grid, (temp_coord_1[0], temp_coord_1[1]), \
                (temp_coord_2[0], temp_coord_2[1]), (self._agent_color_list[ind]), 2)

            temp_file_name = "Grid_with_Path_of_Agent_{}_.png".format(ind)
            cv2.imwrite(os.path.join(self._path_to_save_agents_path, temp_file_name), temp_grid)

            ind += 1


    def run(self):
        
        if self._verbose:
            print("-----------------------------------------------------------")
            print("--------------------REGION-EXPLORATION---------------------")
            print("-----------------------------------------------------------\n")
        
        self._get_nodes_to_explore()
        self._reach_region_and_explore()
        self._draw_path_of_all_agents_on_separate_grid()

        return self._mission_complete


    def _temp_manual_control(self):
        pass
        # grid_mapper = Mapper(global_grid=temp_occupancy_grid_with_obs)

        # display = copy.copy(temp_occupancy_grid_with_obs)
        # font = cv2.FONT_HERSHEY_SIMPLEX

        # while(1):

        #     display = copy.copy(temp_occupancy_grid_with_obs)

        #     for i in range(self._no_of_agents):

        #         temp_pos = agenthandler.get_pos_of_agent(i)

        #         temp_pos_x = temp_pos['x']
        #         temp_pos_y = temp_pos['y']

        #         if i == 0:
        #             color_b,color_g,color_r = 255,0,0
        #         elif i == 1:
        #             color_b,color_g,color_r = 0,255,0
        #         else:
        #             color_b,color_g,color_r = 0,0,255

        #         cv2.circle(temp_occupancy_grid_with_obs, (temp_pos_x, temp_pos_y), 1, (color_b,color_g,color_r), -1)

        #         img = cv2.ellipse(display,(temp_pos_x,temp_pos_y),(10,10),0,15,345,(color_b,color_g,color_r),-1)

        #         grid_mapper.map_grid(agent_no=i, agent_pos=temp_pos)
            
        #     temp_img = cv2.resize(img, (512, 512))

        #     mapped_grid = grid_mapper.get_mapped_grid()
            
        #     temp_mapped_img = cv2.resize(mapped_grid, (512, 512))

        #     numpy_horizontal = np.hstack((temp_img, temp_mapped_img))

        #     cv2.imshow('image', numpy_horizontal)
        #     k = cv2.waitKey(1) & 0xFF

        #     if k == ord('q'):
        #         break
        #     if k == ord('1'):
        #         agent_number = 0
        #     if k == ord('2'):
        #         agent_number = 1
        #     if k == ord('3'):
        #         agent_number = 2
        #     if k == ord('w'):
        #         agenthandler.move_agent(agent_number, 0, -5)
        #     if k == ord('s'):
        #         agenthandler.move_agent(agent_number, 0, 5)
        #     if k == ord('a'):
        #         agenthandler.move_agent(agent_number, -5, 0)
        #     if k == ord('d'):
        #         agenthandler.move_agent(agent_number, 5, 0)


        # cv2.destroyAllWindows()
        # # grid_mapper.show_mapped_grid()

        # # cv2.imshow('Occupancy_grid',img)
        # # cv2.waitKey(0)
        # # cv2.destroyAllWindows()