#!usr/bin/env python

import copy
import time
import math

import numpy as np

from config.Config import Config
from exploration.explorer import Explorer
from agent.agent_handler import AgentHandler
from grid_world_generator.grid_world import GridWorld
from region_assignment.k_mean_clustring import KMeanClustring
from utils.util_functions import get_cost_matrix, stateCoordsToName
from utils.graph import get_closest_vertex_coords_on_graph_from_pos
from region_assignment.hungarian_region_assignment import HungarianRegionAssignment
from occupancy_grid_generator.occupancy_grid_generator import OccupancyGridGenerator


def occupancy_grid_generator():
    
    occupancy_grid = OccupancyGridGenerator()
    occupancy_grid.generate_occupancy_grid()

    # occupancy_grid.show_occupancy_grid_without_obs()
    # occupancy_grid.show_occupancy_grid_with_obs()
    
    temp_occupancy_grid_without_obs = occupancy_grid.get_occupancy_grid_without_obs()
    temp_occupancy_grid_with_obs = occupancy_grid.get_occupancy_grid_with_obs()
    
    return temp_occupancy_grid_without_obs, temp_occupancy_grid_with_obs


def grid_world(temp_occupancy_grid_without_obs):
    
    X_DIM = int(Config.GRID_WIDTH/Config.EDGE_COST)
    Y_DIM = int(Config.GRID_LEN/Config.EDGE_COST)
    print("Edge Cost:", Config.EDGE_COST, ", Width:", X_DIM, ", Height:", \
          Y_DIM, ", Sensor Range:", Config.SENSOR_RANGE, "\n")

    graph_list = []

    for i in range(Config.NO_OF_AGENTS):

        graph_list.append(GridWorld(X_DIM, Y_DIM, temp_occupancy_grid_without_obs))

        graph_list[i].run()
        # temp_graph = copy.copy(graph.get_graph())
        # graph_list[i].show_nodes_on_occupancy_grid()
        # graph_list[i].show_nodes_and_edges_with_obs_on_occupancy_grid()
        # graph_list[i].show_nodes_and_all_traversable_edges()

    temp_grid_with_nodes = graph_list[-1].get_occupancy_grid_with_nodes()
    temp_grid_with_nodes_and_edges_with_obs = graph_list[-1].get_occupancy_grid_with_nodes_and_edges_with_obs()
    temp_grid_with_nodes_and_all_traversable_edges = graph_list[-1].get_occupancy_grid_with_nodes_and_all_traversable_edges()
    # print(graph.graph['x0y0'])
    temp_graph_list = copy.copy(graph_list)
    
    return temp_graph_list, temp_grid_with_nodes


def k_mean_clustring(temp_occupancy_grid_without_obs):
    
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
    print("Region centroids:\n\n", temp_region_centroids, "\n")
    print("Region color map:\n\n", temp_color_map, "\n")
    
    return temp_region_centroids, temp_color_map, temp_grid_with_regions


def region_assignment(temp_region_centroids, agenthandler, temp_grid_with_regions, temp_graph_list, temp_grid_with_nodes, temp_color_map):

    color_map = [[]] * Config.NO_OF_AGENTS
    goal_pos = [''] * Config.NO_OF_AGENTS

    region_centroids = copy.copy(temp_region_centroids)

    cost_matrix = get_cost_matrix(Config.NO_OF_AGENTS, agenthandler, region_centroids)

    hungaian_region_assignment = HungarianRegionAssignment(cost_matrix, temp_grid_with_regions)

    hungaian_region_assignment.find_regions()
    temp_regions_rows, temp_regions_cols = hungaian_region_assignment.get_regions()

    ind = 0
    
    for el in temp_regions_cols:
        temp_x, temp_y = temp_region_centroids[el]
        color_map[ind] = temp_color_map[el]
        temp_goal_pos = get_closest_vertex_coords_on_graph_from_pos(\
                            temp_graph_list[ind].get_graph(), temp_x, temp_y, Config.EDGE_COST)
        # print(temp_goal_pos)
        goal_pos[ind] = stateCoordsToName(temp_goal_pos[1], temp_goal_pos[0], Config.EDGE_COST)

        ind += 1
        
    print("\nNew Color Map Order:", color_map, "\n")
    # hungaian_region_assignment.show_assigned_regions(agenthandler, temp_region_centroids)
    # hungaian_region_assignment.show_assigned_regions(agenthandler, temp_region_centroids, temp_grid_with_nodes)

    print("Region centroids:\n")
    ind = 1
    for el in region_centroids:
        print("\tRegion:", ind, ", x:", el[0], ", y:", el[1])
        ind += 1

    print("\nAgent Positions:\n")

    for i in range(Config.NO_OF_AGENTS):
        temp_agent_pos = agenthandler.get_pos_of_agent(i)
        print("\tAgent:", i+1, ", x:", temp_agent_pos['x'], ", y:", temp_agent_pos['y'])

    print("\nCost Matrix:\n\n", np.array(cost_matrix, dtype=np.int), "\n")

    ind = 1
    for el in temp_regions_cols:
        print("Region Assigned to Agent: {} is Region {}".format(ind, el))
        ind += 1

    print("\nTotal cost:", int(hungaian_region_assignment.get_total_cost()), "\n")
    
    return goal_pos, color_map


def main():
    
    # ------------------ OCCUPANCY GRID GENERATOR ------------------ #
    
    temp_occupancy_grid_without_obs, temp_occupancy_grid_with_obs = occupancy_grid_generator()
    
    # -------------------------- GRID WORLD ------------------------ #
    
    temp_graph_list, temp_grid_with_nodes = grid_world(temp_occupancy_grid_without_obs)

    # ---------------------- K-MEAN-CLUSTRING ---------------------- #
    
    temp_region_centroids, temp_color_map, temp_grid_with_regions = k_mean_clustring(\
                                                                             temp_occupancy_grid_without_obs)
    
    # ------------------------ AGENT HANDLER ----------------------- #
    
    agenthandler = AgentHandler(temp_graph_list)
    
    # --------------------- REGION ASSIGNMENT ---------------------- #
    
    goal_pos, color_map_new = region_assignment(temp_region_centroids, agenthandler, temp_grid_with_regions, \
                      temp_graph_list, temp_grid_with_nodes, temp_color_map)
    
    # ------------------------- Explorer --------------------------- #
    
    explorer = Explorer(global_grid=temp_occupancy_grid_with_obs, assigned_region_node_names=goal_pos, \
                       graph_list=temp_graph_list, agenthandler=agenthandler, color_map=color_map_new, \
                        grid_with_regions=temp_grid_with_regions, verbose=True)
    explorer.run()
    # -------------------------------------------------------------- #


if __name__ == '__main__':
    main()