#!usr/bin/env python

# import copy
import math

import numpy as np


def l2_distance(coord1_x, coord1_y, coord2_x, coord2_y):
    '''
    Calculates the l2 distance between two world coordinates.
    For detailed explanation, visit this link:
    https://en.wikipedia.org/wiki/Euclidean_distance
    Parameters
    ----------
    coord1_x : float
        x of the first coordinate.
    coord1_y : float
        y of the first coordinate.
    coord2_x : float
        x of the second coordinate.
    coord2_y : float
        y of the second coordinate.
    Returns
    -------
    float
        l2 distance between the two given coordinates.
    '''

    dist_x = coord2_x - coord1_x
    dist_y = coord2_y - coord1_y

    return math.sqrt((dist_x * dist_x) + (dist_y * dist_y))# * 1.113195e5


def get_cost_matrix(no_of_agents, agenthandler, region_centroids):

    cost_matrix = np.zeros(shape=[no_of_agents, no_of_agents])

    for i in range(no_of_agents):
        temp_agent_pos = agenthandler.get_pos_of_agent(i)
        a = np.array([temp_agent_pos['x'], temp_agent_pos['y']])
        for j in range(len(region_centroids)):
            b = np.array(region_centroids[j])
            cost_matrix[i,j] = np.linalg.norm(a-b)

    return cost_matrix


def stateNameToCoords(name, edge_cost):
    # print(name.split('x'))
    # print(name.split('x')[1])
    # print(name.split('x')[1].split('y'))
    # print(int(name.split('x')[1].split('y')[0]))
    # print(int(name.split('x')[1].split('y')[1]))

    val_1 = int(name.split('x')[1].split('y')[0])*edge_cost
    val_2 = int(name.split('x')[1].split('y')[1])*edge_cost

    if val_1 == 0:
      val_1 = edge_cost
    if val_2 == 0:
      val_2 = edge_cost

    return [val_1, val_2]
