#!usr/bin/env python

import copy

import numpy as np

from cv2 import cv2

from occupancy_grid_generator.occupancy_grid_generator import OccupancyGridGenerator
from agent.agent_handler import AgentHandler
from mapping.mapper import Mapper
from config.Config import Config

#############################################################################################

occupancy_grid = OccupancyGridGenerator()

occupancy_grid.generate_occupancy_grid()

# occupancy_grid.show_occupancy_grid_without_obs()
# occupancy_grid.show_occupancy_grid_with_obs()

temp_occupancy_grid_without_obs = occupancy_grid.get_occupancy_grid_without_obs()
temp_occupancy_grid_with_obs = occupancy_grid.get_occupancy_grid_with_obs()

# #############################################################################################

agenthandler = AgentHandler()

# #############################################################################################

grid_mapper = Mapper(global_grid=temp_occupancy_grid_with_obs)

display = copy.copy(temp_occupancy_grid_with_obs)
font = cv2.FONT_HERSHEY_SIMPLEX

while(1):

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

        cv2.circle(temp_occupancy_grid_with_obs, (temp_pos_x, temp_pos_y), 1, (color_b,color_g,color_r), -1)

        img = cv2.ellipse(display,(temp_pos_x,temp_pos_y),(10,10),0,15,345,(color_b,color_g,color_r),-1)

        grid_mapper.map_grid(agent_no=i, agent_pos=temp_pos)
    
    temp_img = cv2.resize(img, (512, 512))

    mapped_grid = grid_mapper.get_mapped_grid()
    
    temp_mapped_img = cv2.resize(mapped_grid, (512, 512))

    numpy_horizontal = np.hstack((temp_img, temp_mapped_img))

    cv2.imshow('image', numpy_horizontal)
    k = cv2.waitKey(1) & 0xFF

    if k == ord('q'):
        break
    if k == ord('1'):
        agent_number = 0
    if k == ord('2'):
        agent_number = 1
    if k == ord('3'):
        agent_number = 2
    if k == ord('w'):
        agenthandler.move_agent(agent_number, 0, -5)
    if k == ord('s'):
        agenthandler.move_agent(agent_number, 0, 5)
    if k == ord('a'):
        agenthandler.move_agent(agent_number, -5, 0)
    if k == ord('d'):
        agenthandler.move_agent(agent_number, 5, 0)


cv2.destroyAllWindows()
# grid_mapper.show_mapped_grid()

# cv2.imshow('Occupancy_grid',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()