from cv2 import cv2

from config.Config import Config
from agent.agent_generator import AgentGenerator


class AgentHandler(object):

    def __init__(self):

        self._no_of_agents = Config.NO_OF_AGENTS
        self._agents = []

        for i in range(self._no_of_agents):

            self._agents.append(AgentGenerator())

        for i in range(self._no_of_agents):
            
            self._agents[i].generate_agent()

    
    def get_pos_of_agent(self, agent_no):

        return self._agents[agent_no].get_agent_pos()

    
    def move_agent(self, agent_no, new_x, new_y):

        self._agents[agent_no].move_agent(new_x, new_y)