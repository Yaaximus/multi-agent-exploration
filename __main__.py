#!usr/bin/env python

import copy

import numpy as np

from cv2 import cv2

from occupancy_grid_generator.occupancy_grid_generator import OccupancyGridGenerator
from agent.agent_handler import AgentHandler
from mapping.mapper import Mapper
from region_assignment.k_mean_clustring import KMeanClustring
from region_assignment.hungarian_region_assignment import HungarianRegionAssignment
from config.Config import Config
from utils.util_functions import get_cost_matrix
from grid_world_generator.grid_world import GridWorld

##############################--OCCUPANCY GRID GENERATOR--#####################################

occupancy_grid = OccupancyGridGenerator()

occupancy_grid.generate_occupancy_grid()

# occupancy_grid.show_occupancy_grid_without_obs()
# occupancy_grid.show_occupancy_grid_with_obs()

temp_occupancy_grid_without_obs = occupancy_grid.get_occupancy_grid_without_obs()
temp_occupancy_grid_with_obs = occupancy_grid.get_occupancy_grid_with_obs()

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

agenthandler = AgentHandler()

# ###########################-----REGION ASSIGNMENT-----##########################################

region_centroids = copy.copy(temp_region_centroids)

cost_matrix = get_cost_matrix(Config.NO_OF_AGENTS, agenthandler, region_centroids)

hungaian_region_assignment = HungarianRegionAssignment(cost_matrix, temp_grid_with_regions)

hungaian_region_assignment.find_regions()
temp_regions_rows, temp_regions_cols = hungaian_region_assignment.get_regions()
# print(hungaian_region_assignment.get_total_cost())

hungaian_region_assignment.show_assigned_regions(agenthandler, region_centroids)

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


# ###########################--------GRID WORLD---------##########################################

EDGE_COST = 40
X_DIM = int(Config.GRID_WIDTH/EDGE_COST)
Y_DIM = int(Config.GRID_LEN/EDGE_COST)
VIEWING_RANGE = Config.SENSOR_RANGE
# print(EDGE_COST, X_DIM, Y_DIM, VIEWING_RANGE)

graph = GridWorld(X_DIM, Y_DIM, EDGE_COST, temp_occupancy_grid_without_obs)

graph.run()
graph.show_nodes_on_occupancy_grid()
graph.show_nodes_and_edges_with_obs_on_occupancy_grid()
graph.show_nodes_and_all_traversable_edges()
# print(graph.graph['x0y0'])


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
