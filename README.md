# PYTHO IMPLEMENTATION OF MULTI AGENT EXPLORATION

## Grid Type: Easy
### Number of Agents: 8
### Number of Unknown Obstacle: 1

![8_Agents_grid_easy](https://user-images.githubusercontent.com/37571161/74771788-9b36c100-52b0-11ea-8e95-44fe7bc0c07b.gif)

## Grid Type: Difficult
### Number of Agents: 4
### Number of Unknown Obstacle: 4

![8_Agents_grid_difficult](https://user-images.githubusercontent.com/37571161/74771883-cde0b980-52b0-11ea-9bd4-aec19c580527.gif)

## Grid Type: Easy 
### Number of Agents: 1
### Number of Unknown Obstacle: 4

![1_Agent_grid_easy](https://user-images.githubusercontent.com/37571161/74771943-ed77e200-52b0-11ea-8ebb-17c4001d9b93.gif)


## RUNNING INSTRUCTION

- Import conda env using the following command:

    <code>conda env create -f env.yml</br></code>
    <code>conda activate multiagentexploration</code>

- Run the code using command:

    <code>python multi_agent_exploration.py</code>

## REQUIRED PACKAGES

- Anaconda

    <code>https://www.anaconda.com/distribution/</br></code>
    <code>https://docs.anaconda.com/anaconda/install/</code>

## MODULES

- Multi-Agent-Exploration
    - Agent
        - Agent generator
        - Agent handler
    - D-start-lite
        - D start lite
    - Exploration
        - Explorer
    - Grid world generator
        - Grid world
    - Mapping
        - Mapper
    - Occupance grid generator
        - Occupance grid generator
    - Region Assignment
        - Hungarian region assignment
        - K mean clustring
    - utils
        - graph
        - util finctions
    - config
        - Config(default)
        - config(run time configurations)
    - multi-agent-exploration

## FEATURES
In order to use features. open file /config/config.yaml

- N number of agents for exploration
    - no_of_agents = value
    - value must be integer (range:0~8)
- 4 different difficulty levels
    - complexity_level = value
    - value must be one of the following (very_easy, easy, moderate, difficult)
- variable sensor range
    - sensor_range = value
    - value must be in pixels
    - Default: 45 pixels in every direction

## TROUBLESHOOTING

In order to troubleshoot. open file /config/config.yaml

***Change verbose value from False to True***. This will start displaying helpful information on console.

***Change show_results from False to True***. This will start displaying plots.