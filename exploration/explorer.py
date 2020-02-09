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
                 agenthandler, color_map, grid_with_regions, verbose=False):
                
        self._verbose = verbose
        self._mapped_grid = None
        self._color_map = color_map
        self._img_with_agents = None
        self._graph_list = graph_list
        self._global_grid = global_grid
        self._agent_handler = agenthandler
        self._display = copy.copy(global_grid)
        self._known_grid = copy.copy(known_grid)
        self._k_m_list = [0] * Config.NO_OF_AGENTS
        self._grid_with_regions = grid_with_regions
        self._last_pos = [[]] * Config.NO_OF_AGENTS
        self._queue_list = [[]] * Config.NO_OF_AGENTS
        self._s_new_names = [''] * Config.NO_OF_AGENTS
        self._s_last_names = [''] * Config.NO_OF_AGENTS
        self._s_start_names = [''] * Config.NO_OF_AGENTS
        self._avoiding_obs = [False] * Config.NO_OF_AGENTS
        self._s_current_names = [''] * Config.NO_OF_AGENTS
        self._nodes_to_explore = [[]] * Config.NO_OF_AGENTS
        self._grid_mapper = Mapper(global_grid=self._global_grid)
        self._assigned_region_node_names = assigned_region_node_names
        
        temp_coord = [0,0]
        self._mission_stages_list = []
        for i in range(Config.NO_OF_AGENTS):
            self._mission_stages_list.append({'region_reached':False, 'region_explored':False})
            
        for i in range(Config.NO_OF_AGENTS):
            temp_coord = self._agent_handler.get_pos_of_agent(i)
            self._last_pos[i] = temp_coord
            self._s_start_names[i] = stateCoordsToName(temp_coord['y'], temp_coord['x'], Config.EDGE_COST)
            
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

        if self._verbose:
            print("Explorer: START NODE NAMES:\n")
            print(self._s_start_names, "\n")
            print("Explorer: ASSIGNED REGION NODE NAMES:\n")
            print(self._assigned_region_node_names, "\n")
            print("Explorer: Number of Graphs in list:", len(self._graph_list), "\n")

            for i in range(len(self._s_current_names)):
                pos_coords = stateNameToCoords(self._s_current_names[i], Config.EDGE_COST)

                print("Explorer: Node Name: {} , Node Position: {}.".format(self._s_current_names[i], pos_coords))

            print("")
            
    
    def _update_display(self):
        
        self._display = copy.copy(self._global_grid)
        
        for i in range(Config.NO_OF_AGENTS):
            
            temp_pos = self._agent_handler.get_pos_of_agent(i)
            temp_pos_x = temp_pos['x']
            temp_pos_y = temp_pos['y']
            
            self._update_agents_on_map(i, temp_pos_x, temp_pos_y)
            self._map_world(i, temp_pos)
        
        temp_img_with_agents = cv2.resize(self._img_with_agents, (512, 512))
        self._mapped_grid = self._grid_mapper.get_mapped_grid()
        temp_mapped_img = cv2.resize(self._mapped_grid, (512, 512))
        
        return np.hstack((temp_img_with_agents, temp_mapped_img))

    
    def _update_agents_on_map(self, i, temp_pos_x, temp_pos_y):
        
        if i == 0:
            color_b,color_g,color_r = 255,0,0
        elif i == 1:
            color_b,color_g,color_r = 0,255,0
        else:
            color_b,color_g,color_r = 0,0,255

        self._img_with_agents = cv2.ellipse(self._display,(temp_pos_x,temp_pos_y),\
                                            (10,10),0,15,345,(color_b,color_g,color_r),-1)
    
    
    def _map_world(self, i, temp_pos):
        
        self._graph_list[i], list_of_node_to_remove = self._grid_mapper.map_grid(agent_no=i, agent_pos=temp_pos, \
                                                                                agent_last_pos=self._last_pos[i], graph=self._graph_list[i])
        
        if len(list_of_node_to_remove)>0:
            for el in list_of_node_to_remove:
                if el in self._nodes_to_explore[i]:
                    self._nodes_to_explore[i].remove(str(el))
        
    
    def _get_nodes_to_explore(self):
        
        for i in range(Config.NO_OF_AGENTS):
            
            temp_list = []
            print("Explorer:_get_nodes_to_explore: Agent {} Color Map:{}".format(i, self._color_map[i]))
            for el in self._graph_list[i].graph:
                temp_coords = stateNameToCoords(el, Config.EDGE_COST)
                if np.all(self._grid_with_regions[temp_coords[0], temp_coords[1]] == self._color_map[i], axis=-1):
                    temp_list.append(el)
                    # print(i, el, self._grid_with_regions[temp_coords[0], temp_coords[1]], self._color_map[i])
            self._nodes_to_explore[i] = copy.copy(temp_list)
            # print("Nodes to explore by Agent: {} are {}.".format(i, self._nodes_to_explore[i]))
        print("")
            
            
    def _get_closest_traversable_node(self, i):
        
        temp_dist = math.inf
        temp_current_coords = stateNameToCoords(self._s_current_names[i], Config.EDGE_COST)
        nodes_to_remove = []
        # print("S Current:", self._s_current_names[i], ", Current Node Coord:", temp_current_coords)
        for el in self._nodes_to_explore[i]:
            temp_coords = stateNameToCoords(el, Config.EDGE_COST)
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
        # print("Closest traversable Node Name:{}".format(stateCoordsToName(x, y, Config.EDGE_COST)))
        return stateCoordsToName(x, y, Config.EDGE_COST)


    def _explore_assigned_region(self, i):
        
        self._s_new_names[i] = self._get_closest_traversable_node(i)
        # print("Explorer:_explore_assigned_region: Agent:", i, ", S_current:", self._s_current_names[i], ", S_new", self._s_new_names[i])
        if check_if_no_obs_bw_nodes(self._s_current_names[i], self._s_new_names[i], self._global_grid):
            return True
        else:
            self._avoiding_obs[i] = True
            # print("Explorer:_explore_assigned_region: Obstalce b/w node.")
            X_DIM = int(Config.GRID_WIDTH/Config.EDGE_COST)
            Y_DIM = int(Config.GRID_LEN/Config.EDGE_COST)
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
        
        temp_current_coords = stateNameToCoords(self._s_current_names[i], Config.EDGE_COST)
        temp_new_coords = stateNameToCoords(self._s_new_names[i], Config.EDGE_COST)
        temp_dist = l2_distance(temp_current_coords[0], temp_current_coords[1], \
                                    temp_new_coords[0], temp_new_coords[1])
        
        if temp_dist > Config.EDGE_COST:
            temp_dist = math.inf
            for neighbor in self._graph_list[i].graph[self._s_current_names[i]].children:
                # print("Neighbor Name:", neighbor, "Distance from ")
                neighbor_coords = stateNameToCoords(neighbor, Config.EDGE_COST)
                new_temp_dist = l2_distance(temp_new_coords[0], temp_new_coords[1], neighbor_coords[0], neighbor_coords[1])

                if new_temp_dist < temp_dist:
                    if check_if_no_obs_bw_nodes(self._s_current_names[i], neighbor, self._global_grid):
                        x = neighbor_coords[0]
                        y = neighbor_coords[1]
                        temp_dist = new_temp_dist

            self._s_new_names[i] = stateCoordsToName(x, y, Config.EDGE_COST)
            
    
    def _reach_region_and_explore(self):
        
        # TODO: Incase obs on region assigned road handle this situation
        status = [[]] * Config.NO_OF_AGENTS
        mission_complete = False
        while not mission_complete:
            h_stacked_images = self._update_display()
            cv2.imshow('MULTI AGENT EXPLORER SIMULATOR', h_stacked_images)
            k = cv2.waitKey(1) & 0xFF

            if k == ord('q'):
                print("Mission Aborted!")
                break

            # time.sleep(0.5)
            time.sleep(0.1)

            count = 0
            for el in self._mission_stages_list:
                if el['region_reached'] and el['region_explored']:
                    count += 1
                if count == Config.NO_OF_AGENTS:
                    mission_complete = True
                    print("\nAll agents have Explored Assigned Regions.", "\n")
                    print("Agent's Current Node Names: ", self._s_new_names, "\n")

            for i in range(len(self._graph_list)):
                if not self._mission_stages_list[i]['region_reached']:
                    self._s_new_names[i], self._k_m_list[i] = moveAndRescan(self._graph_list[i], \
                        self._queue_list[i], self._s_current_names[i], Config.SENSOR_RANGE, self._k_m_list[i],i)
                
                if self._mission_stages_list[i]['region_reached'] and not self._mission_stages_list[i]['region_explored']:
                    
                    if not self._avoiding_obs[i]:
                        status[i] = self._explore_assigned_region(i)
                    
                    if status and not self._avoiding_obs[i]:
                        pass
                    else:
                        self._graph_list[i], self._queue_list[i], self._k_m_list[i] = status[i]
                
                        self._s_new_names[i], self._k_m_list[i] = runTimeRescanAndMove(self._graph_list[i], \
                            self._queue_list[i], self._s_current_names[i], Config.SENSOR_RANGE, self._k_m_list[i],i)

                        if self._s_new_names[i] in self._nodes_to_explore[i] or self._s_new_names[i] == 'goal' or self._s_new_names[i] == None:
                            # print("Explorer:_reach_region_and_explore: Exiting...")
                            self._avoiding_obs[i] = False
                            
                if self._s_new_names[i] != 'goal' and self._s_new_names[i] != None:
                    
                    temp_new_coords = stateNameToCoords(self._s_new_names[i], Config.EDGE_COST)
                    if np.all(self._global_grid[temp_new_coords[0], temp_new_coords[1]] == [255, 255, 255], axis=-1):
                        
                        self._check_max_movement_constraint(i)
                        self._last_pos[i] = stateNameToCoords(self._s_current_names[i], Config.EDGE_COST)
                        self._s_current_names[i] = self._s_new_names[i]
                        # print("Explorer:_reach_region_and_explore: s current updated", i, self._s_current_names[i])
                        temp_coord = stateNameToCoords(self._s_current_names[i], Config.EDGE_COST)
                        self._agent_handler.set_pos_of_agent(i, temp_coord[1], temp_coord[0])
                    
                if self._s_current_names[i] in self._nodes_to_explore[i]:
                    self._nodes_to_explore[i].remove(str(self._s_current_names[i]))

                if self._s_current_names[i] == self._assigned_region_node_names[i]:
                    if not self._mission_stages_list[i]['region_reached']:
                        self._mission_stages_list[i]['region_reached'] = True
                        print("Agent: {} has reached assigned region.".format(i))
                    
                if len(self._nodes_to_explore[i]) == 0:
                    if not self._mission_stages_list[i]['region_explored']:
                        self._mission_stages_list[i]['region_explored'] = True
                        print("Agent: {} has explored assigned region.".format(i))
                        
                temp_coords = stateNameToCoords(self._s_current_names[i], Config.EDGE_COST)
                if np.all(self._global_grid[temp_coords[0], temp_coords[1]] == [0, 0, 0], axis=-1):
                    print("On An Obstacle :-P Not Possible.........")
                    while(1):
                        cv2.imshow('MULTI AGENT EXPLORER SIMULATOR', h_stacked_images)
                        k = cv2.waitKey(1) & 0xFF

                        if k == ord('q'):
                            break

                        time.sleep(0.5)

        while(1):
            if k == ord('q'):
                break
            else:
                cv2.imshow('MULTI AGENT EXPLORER SIMULATOR', h_stacked_images)
                k = cv2.waitKey(1) & 0xFF
                time.sleep(1)

        print("Shutting Down...", "\n")
        cv2.destroyAllWindows()
        print("Shutting Down Successfull.\n")


    def run(self):
        
        self._get_nodes_to_explore()
        self._reach_region_and_explore()


    def _temp_manual_control(self):
        pass
        # grid_mapper = Mapper(global_grid=temp_occupancy_grid_with_obs)

        # display = copy.copy(temp_occupancy_grid_with_obs)
        # font = cv2.FONT_HERSHEY_SIMPLEX

        # while(1):

        #     display = copy.copy(temp_occupancy_grid_with_obs)

        #     for i in range(Config.NO_OF_AGENTS):

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