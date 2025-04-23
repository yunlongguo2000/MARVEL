import os
import matplotlib.pyplot as plt
from skimage import io
from skimage.measure import block_reduce
from copy import deepcopy
import numpy as np
from shapely.geometry import Point, Polygon
from skimage.draw import polygon as sk_polygon


from utils.sensor import sensor_work_heading
from parameter import *
from utils.utils import *


class Env:
    def __init__(self, episode_index, fov, sensor_range, plot=False):
        self.episode_index = episode_index
        self.plot = plot
        self.ground_truth, initial_cell = self.import_ground_truth(episode_index)
        self.ground_truth_size = np.shape(self.ground_truth)  # cell
        self.cell_size = CELL_SIZE  # meter

        self.robot_belief = np.ones(self.ground_truth_size) * 127
        self.belief_origin_x = -np.round(initial_cell[0] * self.cell_size, 1)   # meter
        self.belief_origin_y = -np.round(initial_cell[1] * self.cell_size, 1)  # meter

        self.sensor_range = sensor_range  # meter
        self.explored_rate = 0

        self.fov = fov
        self.done = False

        self.robot_belief = sensor_work_heading(initial_cell, self.sensor_range / self.cell_size, self.robot_belief,
                                        self.ground_truth, 0, 360)
        self.belief_info = MapInfo(self.robot_belief, self.belief_origin_x, self.belief_origin_y, self.cell_size)

        free, _ = get_updating_node_coords(np.array([0.0, 0.0]), self.belief_info)
        choice = np.random.choice(free.shape[0], N_AGENTS, replace=False)
        starts = free[choice]
        self.robot_locations = np.array(starts)

        robot_cells = get_cell_position_from_coords(self.robot_locations, self.belief_info).reshape(-1, 2)
        self.angles =  np.random.uniform(0, 360, size=N_AGENTS)   # Intialise the robot heading
        # World frame, origin pointing right
        if N_AGENTS == 1:
            self.robot_belief = sensor_work_heading(robot_cells, self.sensor_range / self.cell_size, self.robot_belief,
                                            self.ground_truth, self.angles[0], self.fov)
        else:
            for i, robot_cell in enumerate(robot_cells):
                self.robot_belief = sensor_work_heading(robot_cell, self.sensor_range / self.cell_size, self.robot_belief,
                                                self.ground_truth, self.angles[i], self.fov)
        self.old_belief = deepcopy(self.robot_belief)
        self.global_frontiers = get_frontier_in_map(self.belief_info)

        self.ground_truth_info = MapInfo(self.ground_truth, self.belief_origin_x, self.belief_origin_y, self.cell_size)

        if self.plot:
            self.frame_files = []

    def import_ground_truth(self, episode_index):
        map_dir = f'maps_medium'
        map_list = os.listdir(map_dir)
        map_index = episode_index % np.size(map_list)
        ground_truth = (io.imread(map_dir + '/' + map_list[map_index], 1)).astype(int)

        ground_truth = block_reduce(ground_truth, 2, np.min)

        robot_cell = np.array(np.nonzero(ground_truth == 208))
        robot_cell = np.array([robot_cell[1, 10], robot_cell[0, 10]])

        ground_truth = (ground_truth > 150) | ((ground_truth <= 80) & (ground_truth >= 50))
        ground_truth = ground_truth * 254 + 1

        return ground_truth, robot_cell

    def update_robot_location(self, robot_location):
        self.robot_location = robot_location
        self.robot_cell = np.array([round((robot_location[0] - self.belief_origin_x) / self.cell_size),
                                    round((robot_location[1] - self.belief_origin_y) / self.cell_size)])

    def update_robot_belief(self, robot_cell, heading):
        self.robot_belief = sensor_work_heading(robot_cell, round(self.sensor_range / self.cell_size), self.robot_belief,
                                        self.ground_truth, heading, self.fov)

    def calculate_team_reward(self):
        reward = 0

        global_frontiers = get_frontier_in_map(self.belief_info)
        if len(global_frontiers) == 0:
            delta_num = len(self.global_frontiers)
        else:
            observed_frontiers = self.global_frontiers - global_frontiers
            delta_num = len(observed_frontiers)

        reward += delta_num / (self.sensor_range * 3.14 // FRONTIER_CELL_SIZE) 

        self.global_frontiers = global_frontiers
        self.old_belief = deepcopy(self.robot_belief)

        return reward

    def check_done(self):
        if np.sum(self.ground_truth == 255) - np.sum(self.robot_belief == 255) <= 250:
            self.done = True

    def evaluate_exploration_rate(self):
        self.explored_rate = np.sum(self.robot_belief == 255) / np.sum(self.ground_truth == 255)

    def step(self, next_waypoint, agent_id):
        self.evaluate_exploration_rate()
        self.robot_locations[agent_id] = next_waypoint
        reward = 0
        cell = get_cell_position_from_coords(next_waypoint, self.belief_info)
        self.update_robot_belief(cell)

        return reward
    
    def final_sim_step(self, next_waypoint, agent_id):
        self.evaluate_exploration_rate()
        self.robot_locations[agent_id] = next_waypoint


    def heading_to_vector(self, heading, length=25):
        # Convert heading to vector
        if isinstance(heading, (list, np.ndarray)):
            heading = heading[0]
        heading_rad = np.radians(heading)
        return np.cos(heading_rad) * length, np.sin(heading_rad) * length
    
    def create_sensing_mask(self, location, heading):
        mask = np.zeros_like(self.ground_truth)

        location_cell = get_cell_position_from_coords(location, self.belief_info)
        # Create a Point for the robot's location
        robot_point = Point(location_cell)
        # heading = heading*(360/self.num_angles)

        # Calculate the angles for the sector
        start_angle = (heading - self.fov / 2 + 360) % 360
        end_angle = (heading + self.fov / 2) % 360

        # Create points for the sector
        sector_points = [robot_point]
        if start_angle <= end_angle:
            angle_range = np.linspace(start_angle, end_angle, 20)
        else:
            angle_range = np.concatenate([np.linspace(start_angle, 360, 10), np.linspace(0, end_angle, 10)])
        for angle in angle_range: 
            x = location_cell[0] + self.sensor_range/CELL_SIZE * np.cos(np.radians(angle))
            y = location_cell[1] + self.sensor_range/CELL_SIZE * np.sin(np.radians(angle))
            sector_points.append(Point(x, y))
        sector_points.append(robot_point) 

        sector = Polygon(sector_points)

        x_coords, y_coords = sector.exterior.xy
        y_coords = np.rint(y_coords).astype(int)
        x_coords = np.rint(x_coords).astype(int)
        rr, cc = sk_polygon(
                [int(round(y)) for y in y_coords],
                [int(round(x)) for x in x_coords],
                shape=mask.shape
            )
        
        free_connected_map = get_free_and_connected_map(location, self.belief_info)

        mask[rr, cc] = (free_connected_map[rr, cc] == free_connected_map[location_cell[1], location_cell[0]])
       
        return mask
    
    def calculate_overlap_reward(self, all_robots_locations, robot_headings_list):
        all_robot_sensing_mask = []
        ## Robot heading list in degrees
        for robot_location, robot_heading in zip(all_robots_locations, robot_headings_list):
            robot_sensing_mask = self.create_sensing_mask(robot_location, robot_heading)
            all_robot_sensing_mask.append(robot_sensing_mask)
        
        total_mask = np.sum(all_robot_sensing_mask, axis=0)
        total_sensing_area = np.sum(total_mask > 0)
        total_overlap_area = np.sum(total_mask > 1)
        overlap_reward = (total_sensing_area - total_overlap_area) / total_sensing_area     
        
        return overlap_reward



