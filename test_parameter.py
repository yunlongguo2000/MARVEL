"""
Configuration parameters for marvel testing.

This file defines various parameters for:
- Model loading and testing configuration
- Simulation parameters
- Heading and navigation settings
- Map representation and resolution
- Sensor and utility thresholds
- Network and graph parameters
- GPU usage configuration

"""
TEST_SET = 'maps_test'
LOAD_FOLDER_NAME = 'MARVEL'
load_path = f'load_model/{LOAD_FOLDER_NAME}'
gifs_path = f'results/gifs/{LOAD_FOLDER_NAME}'
LOAD_MODEL = True  
SAVE_IMG_GAP = 50
SAVE_GIFS = True
GREEDY = True
NUM_RUN = 1
NUM_TEST = 100

# Sim parameters
USE_CONTINUOUS_SIM = True
NUM_SIM_STEPS = 6
MAX_EPISODE_STEP = 128
VELOCITY = 1
YAW_RATE = 35 # in degrees

# Heading parameters
NUM_ANGLES_BIN = 36
NUM_HEADING_CANDIDATES = 3

# Map and planning resolution
CELL_SIZE = 0.4  # meter
NODE_RESOLUTION = 4.0  # meter
FRONTIER_CELL_SIZE = 2 * CELL_SIZE

# Map representation
FREE = 255
OCCUPIED = 1
UNKNOWN = 127

# Sensor and utility range
MIN_UTILITY = 1

# Updating map range w.r.t the robot
UPDATING_MAP_SIZE = 15 * NODE_RESOLUTION

# Testing parameters
NUM_META_AGENT = 10
INITIAL_EXPLORED_RATE = 0.90

# Network parameters
NODE_INPUT_DIM = 6
EMBEDDING_DIM = 128

# Graph parameters
NUM_NODE_NEIGHBORS = 5
K_SIZE = NUM_NODE_NEIGHBORS**2   # the number of neighboring nodes

# GPU usage
USE_GPU = False  
NUM_GPU = 0
