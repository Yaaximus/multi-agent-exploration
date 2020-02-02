#!usr/bin/env python

import copy
import time

import numpy as np

from cv2 import cv2
from config.Config import Config
from mapping.mapper import Mapper
from d_star_lite.d_star_lite import *
from agent.agent_handler import AgentHandler
from utils.util_functions import get_cost_matrix
from utils.util_functions import stateCoordsToName
from grid_world_generator.grid_world import GridWorld
from region_assignment.k_mean_clustring import KMeanClustring
from utils.graph import get_closest_vertex_coords_on_graph_from_pos
from region_assignment.hungarian_region_assignment import HungarianRegionAssignment
from occupancy_grid_generator.occupancy_grid_generator import OccupancyGridGenerator

##############################--OCCUPANCY GRID GENERATOR--#####################################

occupancy_grid = OccupancyGridGenerator()

occupancy_grid.generate_occupancy_grid()

# occupancy_grid.show_occupancy_grid_without_obs()
# occupancy_grid.show_occupancy_grid_with_obs()

temp_occupancy_grid_without_obs = occupancy_grid.get_occupancy_grid_without_obs()
temp_occupancy_grid_with_obs = occupancy_grid.get_occupancy_grid_with_obs()

# ###########################--------GRID WORLD---------##########################################

X_DIM = int(Config.GRID_WIDTH/Config.EDGE_COST)
Y_DIM = int(Config.GRID_LEN/Config.EDGE_COST)
VIEWING_RANGE = Config.SENSOR_RANGE
print("Edge Cost:", Config.EDGE_COST, ", Width:", X_DIM, ", Height:", \
      Y_DIM, ", Viewing Range:", VIEWING_RANGE)

graph_list = []

for i in range(Config.NO_OF_AGENTS):

    graph_list.append(GridWorld(X_DIM, Y_DIM, Config.EDGE_COST, temp_occupancy_grid_without_obs))

    graph_list[-1].run()
    #     temp_graph = copy.copy(graph.get_graph())
#     graph_list[-1].show_nodes_on_occupancy_grid()
graph_list[-1].show_nodes_on_occupancy_grid()
graph_list[-1].show_nodes_and_edges_with_obs_on_occupancy_grid()
graph_list[-1].show_nodes_and_all_traversable_edges()
temp_grid_with_nodes = graph_list[-1].get_occupancy_grid_with_nodes()
temp_grid_with_nodes_and_edges_with_obs = graph_list[-1].get_occupancy_grid_with_nodes_and_edges_with_obs()
temp_grid_with_nodes_and_all_traversable_edges = graph_list[-1].get_occupancy_grid_with_nodes_and_all_traversable_edges()
# print(graph.graph['x0y0'])
temp_graph_list = copy.copy(graph_list)

# ###########################------K-MEAN-CLUSTRING------#######################################

regions = KMeanClustring(temp_occupancy_grid_without_obs)

regions.find_regions()

temp_regions_xy_points = regions.get_regions_xy_points()
temp_region_centroids = regions.get_centroids()
temp_color_map = regions.get_color_map()
temp_grid_with_regions = copy.copy(regions.get_grid_with_regions())

# regions.show_regions_with_centroids()
# regions.show_regions()

# ind = 0
# for el in temp_regions_xy_points:
#     print("Region", ind, "indices:", el)
#     ind += 1
print("Region centroids:", temp_region_centroids)
print("Region color map:", temp_color_map)

# ###########################-------AGENT HANDLER-------##########################################

agenthandler = AgentHandler(graph_list)

# ###########################-----REGION ASSIGNMENT-----##########################################

goal_pos = [''] * Config.NO_OF_AGENTS

region_centroids = copy.copy(temp_region_centroids)

cost_matrix = get_cost_matrix(Config.NO_OF_AGENTS, agenthandler, region_centroids)

hungaian_region_assignment = HungarianRegionAssignment(cost_matrix, temp_grid_with_regions)

hungaian_region_assignment.find_regions()
temp_regions_rows, temp_regions_cols = hungaian_region_assignment.get_regions()

ind = 0
for el in temp_regions_cols:
    temp_x, temp_y = temp_region_centroids[el]
    
    temp_goal_pos = get_closest_vertex_coords_on_graph_from_pos(\
                        graph_list[ind].get_graph(), temp_x, temp_y, Config.EDGE_COST)
#     print(temp_goal_pos)
    goal_pos[ind] = stateCoordsToName(temp_goal_pos[1], temp_goal_pos[0], Config.EDGE_COST)
    
    ind += 1

hungaian_region_assignment.show_assigned_regions(agenthandler, temp_region_centroids)
hungaian_region_assignment.show_assigned_regions(agenthandler, temp_region_centroids, temp_grid_with_nodes)

print("Region centroids:\n")
ind = 1
for el in region_centroids:
    print("\tRegion:", ind, ", x:", el[0], ", y:", el[1])
    ind += 1
    
print("\nAgent Positions:\n")

for i in range(Config.NO_OF_AGENTS):
    temp_agent_pos = agenthandler.get_pos_of_agent(i)
    print("\tAgent:", i+1, ", x:", temp_agent_pos['x'], ", y:", temp_agent_pos['y'])
    
print("\nCost Matrix:\n\n", cost_matrix, "\n")

ind = 1
for el in temp_regions_cols:
    print("Region Assigned to Agent: {} is Region {}".format(ind, el))
    ind += 1
    
print("\nTotal cost:", hungaian_region_assignment.get_total_cost())

# ################################################################################################

# grid_mapper = Mapper(global_grid=temp_occupancy_grid_without_obs)
grid_mapper = Mapper(global_grid=temp_occupancy_grid_with_obs)

display = copy.copy(temp_occupancy_grid_without_obs)
font = cv2.FONT_HERSHEY_SIMPLEX

s_start_names = [''] * Config.NO_OF_AGENTS
s_current_names = [''] * Config.NO_OF_AGENTS
s_goal_names = copy.copy(goal_pos)
k_m_list = [0] * Config.NO_OF_AGENTS
s_new_names = [''] * Config.NO_OF_AGENTS
s_last_names = s_start_names
queue_list = [[]] * Config.NO_OF_AGENTS
goal_reached_status_list = [False] * Config.NO_OF_AGENTS
graph_list = copy.copy(temp_graph_list)
temp_coord = [0,0]

for i in range(Config.NO_OF_AGENTS):
    
    temp_coord = agenthandler.get_pos_of_agent(i)
    s_start_names[i] = stateCoordsToName(temp_coord['y'], temp_coord['x'], Config.EDGE_COST)

print("START NODE NAMES:")
print(s_start_names, "\n")
print("GOAL NODE NAMES:")
print(s_goal_names, "\n")
print("Number of Graphs in list:", len(graph_list), "\n")

for i in range(len(graph_list)):
    graph_list[i].setStart(s_start_names[i])
    graph_list[i].setGoal(s_goal_names[i])
    graph_list[i], queue_list[i], k_m_list[i] = initDStarLite(graph_list[i], queue_list[i], \
                                                  s_start_names[i], s_goal_names[i], k_m_list[i])

s_current_names = s_start_names
for i in range(len(s_current_names)):
    pos_coords = stateNameToCoords(s_current_names[i], Config.EDGE_COST)

    print(s_current_names[i], pos_coords)

print("")

mission_complete = False

while not mission_complete:
    
    display = copy.copy(temp_occupancy_grid_with_obs)
    
    for i in range(Config.NO_OF_AGENTS):

        temp_pos = agenthandler.get_pos_of_agent(i)

        temp_pos_x = temp_pos['x']
        temp_pos_y = temp_pos['y']

        if i == 0:
            color_b,color_g,color_r = 255,0,0
        elif i == 1:
            color_b,color_g,color_r = 0,255,0
        else:
            color_b,color_g,color_r = 0,0,255

        img = cv2.ellipse(display,(temp_pos_x,temp_pos_y),(10,10),0,15,345,(color_b,color_g,color_r),-1)
        
        graph_list[i] = grid_mapper.map_grid(agent_no=i, agent_pos=temp_pos, graph=graph_list[i])

    temp_img = cv2.resize(img, (512, 512))

    mapped_grid = grid_mapper.get_mapped_grid()

    temp_mapped_img = cv2.resize(mapped_grid, (512, 512))

    numpy_horizontal = np.hstack((temp_img, temp_mapped_img))

    cv2.imshow('image', numpy_horizontal)
    k = cv2.waitKey(1) & 0xFF
    
    if k == ord('q'):
        print("Mission Failed!")
        break

    time.sleep(0.5)
    
    count = 0
    for el in goal_reached_status_list:
        if el is True:
            count += 1
        if count == Config.NO_OF_AGENTS:
            mission_complete = True
            print("Mission Complete.", "\n")
            for el in s_new_names:
                print(el)
    
    for i in range(len(graph_list)):
        
        if goal_reached_status_list[i] is False:
            s_new_names[i], k_m_list[i] = moveAndRescan(graph_list[i], queue_list[i], s_current_names[i], \
                                                        VIEWING_RANGE, k_m_list[i])
                
        s_current_names[i] = s_new_names[i]
        
#         if i == 0 and goal_reached_status_list[i] == False:
#             print(i, s_new_names[i], k_m_list[i])
#         if i == 1 and goal_reached_status_list[i] == False:
#             print("\t\t", i, s_new_names[i], k_m_list[i])
#         if i == 2 and goal_reached_status_list[i] == False:
#             print("\t\t\t\t", i, s_new_names[i], k_m_list[i])
        
        temp_coord = stateNameToCoords(s_current_names[i], Config.EDGE_COST)
#         print("New Coord:", temp_coord[1], temp_coord[0])
        agenthandler.set_pos_of_agent(i, temp_coord[1], temp_coord[0])
            
        if s_current_names[i] == s_goal_names[i]:
                goal_reached_status_list[i] = True

while(1):
    
    if k == ord('q'):
        break
    else:
        cv2.imshow('image', numpy_horizontal)
        k = cv2.waitKey(1) & 0xFF
        time.sleep(1)

print("Shutting Down...")
cv2.destroyAllWindows()
print("Shutting Down Successfull.")

# ###########################-------GRID MAPPER---------##########################################

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
