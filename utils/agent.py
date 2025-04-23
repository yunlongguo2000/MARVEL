import copy
import time
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
from skimage.draw import polygon as sk_polygon

from utils.utils import *
from parameter import *

class Agent:
    def __init__(self, id, policy_net, fov, heading, sensor_range, node_manager, ground_truth_node_manager, device='cpu', plot=False):
        self.id = id
        self.device = device
        self.plot = plot
        self.policy_net = policy_net
        self.fov = fov
        self.num_prev_headings = 3
        self.sensor_range = sensor_range
        self.num_angles_bin = NUM_ANGLES_BIN
        self.num_heading_candidates = NUM_HEADING_CANDIDATES

        # location and global map
        self.location = None
        self.map_info = None

        # Motion parameters
        self.velocity = VELOCITY
        self.yaw_rate = YAW_RATE

        # map related parameters
        self.cell_size = CELL_SIZE
        self.node_resolution = NODE_RESOLUTION
        self.updating_map_size = UPDATING_MAP_SIZE

        # map and updating map
        self.map_info = None
        self.updating_map_info = None

        # frontiers
        self.frontier = set()

        # node managers
        self.node_manager = node_manager
        self.ground_truth_node_manager = ground_truth_node_manager

        # graph
        self.node_coords, self.utility, self.guidepost, self.occupancy = None, None, None, None
        self.current_index, self.adjacent_matrix, self.neighbor_indices = None, None, None

        self.highest_utility_angles, self.frontier_distribution, self.heading_visited = None, None, None
        self.path_coords = None

        self.travel_dist = 0

          # Heading info
        angle = 0 if heading == 360 else heading
        self.heading = angle

        self.episode_buffer = []
        for i in range(NUM_EPISODE_BUFFER):
            self.episode_buffer.append([])

        if self.plot:
            self.trajectory_x = []
            self.trajectory_y = []


    def update_map(self, map_info):
        # no need in training because of shallow copy
        self.map_info = map_info

    def update_updating_map(self, location):
        self.updating_map_info = self.get_updating_map(location)

    def update_location(self, location):
        if self.location is None:
            dist = 0
        else:
            dist = np.linalg.norm(self.location - location)
        self.travel_dist += dist

        self.location = location

        node = self.node_manager.nodes_dict.find(location.tolist())
        if self.node_manager.nodes_dict.__len__() == 0:
            pass
        else:
            node.data.set_visited(self.heading)
            
        if self.plot:
            self.trajectory_x.append(location[0])
            self.trajectory_y.append(location[1])

    def update_frontiers(self):
        self.frontier = get_frontier_in_map(self.updating_map_info)

    def update_heading(self, heading):
        # Update heading data
        self.heading = heading
        
    def get_updating_map(self, location):
        updating_map_origin_x = (location[0] - self.updating_map_size / 2)
        updating_map_origin_y = (location[1] - self.updating_map_size / 2)

        updating_map_top_x = updating_map_origin_x + self.updating_map_size
        updating_map_top_y = updating_map_origin_y + self.updating_map_size

        min_x = self.map_info.map_origin_x
        min_y = self.map_info.map_origin_y
        max_x = (self.map_info.map_origin_x + self.cell_size * (self.map_info.map.shape[1] - 1))
        max_y = (self.map_info.map_origin_y + self.cell_size * (self.map_info.map.shape[0] - 1))

        if updating_map_origin_x < min_x:
            updating_map_origin_x = min_x
        if updating_map_origin_y < min_y:
            updating_map_origin_y = min_y
        if updating_map_top_x > max_x:
            updating_map_top_x = max_x
        if updating_map_top_y > max_y:
            updating_map_top_y = max_y

        updating_map_origin_x = (updating_map_origin_x // self.cell_size + 1) * self.cell_size
        updating_map_origin_y = (updating_map_origin_y // self.cell_size + 1) * self.cell_size
        updating_map_top_x = (updating_map_top_x // self.cell_size) * self.cell_size
        updating_map_top_y = (updating_map_top_y // self.cell_size) * self.cell_size

        updating_map_origin_x = np.round(updating_map_origin_x, 1)
        updating_map_origin_y = np.round(updating_map_origin_y, 1)
        updating_map_top_x = np.round(updating_map_top_x, 1)
        updating_map_top_y = np.round(updating_map_top_y, 1)

        updating_map_origin = np.array([updating_map_origin_x, updating_map_origin_y])
        updating_map_origin_in_global_map = get_cell_position_from_coords(updating_map_origin, self.map_info)

        updating_map_top = np.array([updating_map_top_x, updating_map_top_y])
        updating_map_top_in_global_map = get_cell_position_from_coords(updating_map_top, self.map_info)

        updating_map = self.map_info.map[
                    updating_map_origin_in_global_map[1]:updating_map_top_in_global_map[1]+1,
                    updating_map_origin_in_global_map[0]:updating_map_top_in_global_map[0]+1]

        updating_map_info = MapInfo(updating_map, updating_map_origin_x, updating_map_origin_y, self.cell_size)

        return updating_map_info

    def update_graph(self, map_info, location):
        self.update_map(map_info)
        self.update_location(location)
        self.update_updating_map(self.location)
        self.update_frontiers()
        self.node_manager.update_graph(self.location,
                                       self.frontier,
                                       self.updating_map_info,
                                       self.map_info)

    def update_planning_state(self, robot_locations):
        self.node_coords, self.utility, self.guidepost, self.occupancy, self.adjacent_matrix, self.current_index, self.neighbor_indices, self.highest_utility_angles, self.frontier_distribution, self.heading_visited, self.path_coords = \
            self.node_manager.get_all_node_graph(self.location, robot_locations)

    def get_observation(self, pad=True):
        node_coords = self.node_coords
        node_utility = self.utility.reshape(-1, 1)
        node_guidepost = self.guidepost.reshape(-1, 1)
        node_occupancy = self.occupancy.reshape(-1, 1)
        node_highest_utility_angles = self.highest_utility_angles.reshape(-1, 1)
        node_frontier_distribution = self.frontier_distribution.reshape(-1, self.num_angles_bin)
        node_heading_visited = self.heading_visited.reshape(-1, self.num_angles_bin)
        current_index = self.current_index
        edge_mask = self.adjacent_matrix
        current_edge = self.neighbor_indices
        n_node = node_coords.shape[0]

        current_node_coords = node_coords[self.current_index]
        all_node_coords = np.concatenate((node_coords[:, 0].reshape(-1, 1) - current_node_coords[0],
                                             node_coords[:, 1].reshape(-1, 1) - current_node_coords[1]),
                                           axis=-1) / UPDATING_MAP_SIZE / 2
        node_utility = node_utility / (2 * self.sensor_range * 3.14 // FRONTIER_CELL_SIZE)
        node_highest_utility_angles = node_highest_utility_angles / 360
        node_frontier_distribution = node_frontier_distribution / ((2 * self.sensor_range * 3.14 // FRONTIER_CELL_SIZE) / self.num_angles_bin)
        node_inputs = np.concatenate((all_node_coords, node_utility, node_guidepost, node_occupancy, node_highest_utility_angles), axis=1)
        node_inputs = torch.FloatTensor(node_inputs).unsqueeze(0).to(self.device)
        all_node_frontier_distribution = torch.Tensor(node_frontier_distribution).unsqueeze(0).to(self.device)
        node_heading_visited = torch.Tensor(node_heading_visited).unsqueeze(0).to(self.device)

        node_padding_mask = torch.zeros((1, 1, n_node), dtype=torch.int16).to(self.device)

        if pad:
            padding = torch.nn.ZeroPad2d((0, 0, 0, NODE_PADDING_SIZE - n_node))
            node_inputs = padding(node_inputs)
            all_node_frontier_distribution = padding(all_node_frontier_distribution)
            node_heading_visited = padding(node_heading_visited)

            node_padding = torch.ones((1, 1, NODE_PADDING_SIZE - n_node), dtype=torch.int16).to(
                self.device)
            node_padding_mask = torch.cat((node_padding_mask, node_padding), dim=-1)

        current_index = torch.tensor([current_index]).reshape(1, 1, 1).to(self.device)

        edge_mask = torch.tensor(edge_mask).unsqueeze(0).to(self.device)

        if pad:
            padding = torch.nn.ConstantPad2d(
                (0, NODE_PADDING_SIZE - n_node, 0, NODE_PADDING_SIZE - n_node), 1)
            edge_mask = padding(edge_mask)

        current_in_edge = np.argwhere(current_edge == self.current_index)[0][0]
        current_edge = torch.tensor(current_edge).unsqueeze(0)
        k_size = current_edge.size()[-1]
        if pad:
            padding = torch.nn.ConstantPad1d((0, K_SIZE - k_size), 0)
            current_edge = padding(current_edge)
        current_edge = current_edge.unsqueeze(-1)
     
        node_neighbor_best_headings, self.neighbor_best_indices = self.compute_best_heading(node_coords, node_frontier_distribution, current_edge)

        edge_padding_mask = torch.zeros((1, 1, k_size), dtype=torch.int16).to(self.device)
        edge_padding_mask[0, 0, current_in_edge] = 1

        if pad:
            padding = torch.nn.ConstantPad1d((0, K_SIZE - k_size), 1)
            edge_padding_mask = padding(edge_padding_mask)

        return [node_inputs, node_padding_mask, edge_mask, current_index, current_edge, edge_padding_mask, all_node_frontier_distribution, node_heading_visited, node_neighbor_best_headings]

    def select_next_waypoint(self, observation, greedy = False):
        _, _, _, _, current_edge, _, _, _, _ = observation
        with torch.no_grad():
            logp = self.policy_net(*observation)

        if greedy:
            action_index = torch.argmax(logp, 1).long()
        else:
            action_index = torch.multinomial(logp.exp(), 1).long().squeeze(1)

        waypoint_index = action_index.item() // self.num_heading_candidates
        next_node_index = current_edge[0, waypoint_index, 0].item()
        heading_index = self.neighbor_best_indices[waypoint_index][action_index.item() % self.num_heading_candidates]
        next_position = self.node_coords[next_node_index]

        return next_position, next_node_index, action_index, heading_index
    
    def compute_best_heading(self, node_coords, frontier_distribution, neighbor_nodes):
        neighbor_best_headings = []
        neighbor_best_indices = []
        neighbor_nodes = list(neighbor_nodes[0])
        for i, neighbor in enumerate(neighbor_nodes):
            node_index = neighbor.item()
            heading_candidates = torch.zeros(self.num_heading_candidates, self.num_angles_bin)
            if (node_index != 0) or (i == 0 and node_index == 0):
                coords = node_coords[node_index]
                node_data = self.node_manager.nodes_dict.find((coords[0], coords[1])).data
                if node_data.utility > 0:
                    node_frontier_distribution = frontier_distribution[node_index]
                    half_fov_size = int((self.fov / 360)*self.num_angles_bin/2)
                    window = np.concatenate((node_frontier_distribution[-half_fov_size:], node_frontier_distribution, node_frontier_distribution[:half_fov_size]))
                    indices = np.arange(len(node_frontier_distribution)) + half_fov_size
                    sum_vector = np.sum(np.take(window, indices.reshape(-1, 1) + np.arange(-half_fov_size, half_fov_size + 1)), axis=1)
                    top_n_indices = np.argsort(-sum_vector)[:self.num_heading_candidates]
                    for i in range(-half_fov_size, half_fov_size+1):
                        indices = (top_n_indices + i) % self.num_angles_bin
                        heading_candidates += F.one_hot(torch.tensor(indices), num_classes=self.num_angles_bin).float()
                else:
                    top_n_indices = np.zeros(3)
                    # Face the robot towards the A* path within 1 bin variance
                    if len(self.path_coords) > 1:
                        next_coords = self.path_coords[1]
                        angle = np.degrees(np.arctan2(next_coords[1] - coords[1],
                                                next_coords[0] - coords[0]) % (2 * np.pi))
                        new_index = int(angle / 360 * self.num_angles_bin) % self.num_angles_bin
                        new_indices = [(new_index + i - self.num_heading_candidates // 2) % self.num_angles_bin for i in range(self.num_heading_candidates)]
                        for l in range(self.num_heading_candidates):
                            heading_candidates[l][int(new_indices[l] - self.fov/2):int(new_indices[l] + self.fov/2)] = 1
                            top_n_indices[l] = new_indices[l]
                    else:
                        for l in range(self.num_heading_candidates):
                            # Make the robot face the direction of neighbor nodes
                            neighbor_list = node_data.neighbor_list[1:]     # First node is self
                            for l in range(self.num_heading_candidates):
                                previous_index = 0
                                if l < len(neighbor_list):
                                    neighbor_coords = neighbor_list[l]                                     # First node is self
                                    angle = np.degrees(np.arctan2(neighbor_coords[1] - coords[1], 
                                                            neighbor_coords[0] - coords[0]) % (2 * np.pi))
                                    new_index = int(angle / 360 * self.num_angles_bin) % self.num_angles_bin
                                    heading_candidates[l][int(new_index-self.fov/2):int(new_index+self.fov/2)] = 1
                                    previous_index = new_index
                                    top_n_indices[l] = new_index
                                else:
                                    heading_candidates[l][previous_index+1] = 1
                                    top_n_indices[l] = previous_index+1
                neighbor_best_headings.append(heading_candidates)
                neighbor_best_indices.append(top_n_indices)
            else:
                neighbor_best_headings.append(heading_candidates)
                neighbor_best_indices.append(np.zeros((1,3)))
        neighbor_best_headings = torch.stack(neighbor_best_headings).unsqueeze(0).to(self.device)
        return neighbor_best_headings, neighbor_best_indices
    
    def check_coords_in_path(self, coords):
        if coords in self.path_coords:
            index = self.path_coords.index(coords)
            next_coord = self.path_coords[index + 1] if index + 1 < len(self.path_coords) else None
            return next_coord
        return  None
    
    def heading_to_vector(self, heading, length=25):
        # Convert heading to vector
        if isinstance(heading, (list, np.ndarray)):
            heading = heading[0]
        heading_rad = np.radians(heading)
        return np.cos(heading_rad) * length, np.sin(heading_rad) * length
    
    def create_sensing_mask(self, location, heading, mask):

        location_cell = get_cell_position_from_coords(location, self.map_info)
        # Create a Point for the robot's location
        robot_point = Point(location_cell)

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
        # Create the sector polygon
        sector = Polygon(sector_points)

        x_coords, y_coords = sector.exterior.xy
        y_coords = np.rint(y_coords).astype(int)
        x_coords = np.rint(x_coords).astype(int)
        rr, cc = sk_polygon(
                [int(round(y)) for y in y_coords],
                [int(round(x)) for x in x_coords],
                shape=mask.shape
            )
        
        free_connected_map = get_free_and_connected_map(location, self.map_info)

        mask[rr, cc] = (free_connected_map[rr, cc] == free_connected_map[location_cell[1], location_cell[0]])
       
        return mask
    
    def calculate_overlap_reward(self, current_robot_location, all_robots_locations, robot_headings_list):
        ## Robot heading list in degrees
        current_sensing_mask = np.zeros_like(self.map_info.map)
        other_robot_sensing_mask = np.zeros_like(self.map_info.map)
        
        for robot_location, robot_heading in zip(all_robots_locations, robot_headings_list):
            if np.array_equal(current_robot_location, robot_location):       
                current_sensing_mask = self.create_sensing_mask(robot_location, robot_heading, current_sensing_mask) 
            else:
                other_robot_sensing_mask = self.create_sensing_mask(robot_location, robot_heading, other_robot_sensing_mask)

        # Keep cell value of 1 only for cells that hold a value of 255 in self.global_map_info.map
        current_free_area_size = np.sum(current_sensing_mask)
        unique_sensing_mask = np.logical_and(current_sensing_mask == 1, other_robot_sensing_mask == 0).astype(int)
        # Compute the number of cells that have a value of 1 in current_sensing_mask and 0 in other_robot_sensing_mask
        current_free_area_not_scanned_size = np.sum(unique_sensing_mask)

        overlap_reward = np.square(current_free_area_not_scanned_size / current_free_area_size)     

        
        return overlap_reward
        
    def save_observation(self, observation):
        node_inputs, node_padding_mask, edge_mask, current_index, current_edge, edge_padding_mask, frontier_distribution, heading_visited, neighbor_best_headings = observation
        self.episode_buffer[0] += node_inputs
        self.episode_buffer[1] += node_padding_mask.bool()
        self.episode_buffer[2] += edge_mask.bool()
        self.episode_buffer[3] += current_index
        self.episode_buffer[4] += current_edge
        self.episode_buffer[5] += edge_padding_mask.bool()
        self.episode_buffer[6] += frontier_distribution
        self.episode_buffer[7] += heading_visited
        self.episode_buffer[38] += neighbor_best_headings

    def save_action(self, action_index):
        self.episode_buffer[8] += action_index.reshape(1, 1, 1)

    def save_reward(self, reward):
        self.episode_buffer[9] += torch.FloatTensor([reward]).reshape(1, 1, 1).to(self.device)

    def save_done(self, done):
        self.episode_buffer[10] += torch.tensor([int(done)]).reshape(1, 1, 1).to(self.device)

    def save_next_observations(self, observation, next_node_index_list):
        self.episode_buffer[11] = copy.deepcopy(self.episode_buffer[0])[1:]
        self.episode_buffer[12] = copy.deepcopy(self.episode_buffer[1])[1:]
        self.episode_buffer[13] = copy.deepcopy(self.episode_buffer[2])[1:]
        self.episode_buffer[14] = copy.deepcopy(self.episode_buffer[3])[1:]
        self.episode_buffer[15] = copy.deepcopy(self.episode_buffer[4])[1:]
        self.episode_buffer[16] = copy.deepcopy(self.episode_buffer[5])[1:]
        self.episode_buffer[17] = copy.deepcopy(self.episode_buffer[6])[1:]
        self.episode_buffer[18] = copy.deepcopy(self.episode_buffer[7])[1:]
        self.episode_buffer[36] = copy.deepcopy(self.episode_buffer[35])[1:]
        self.episode_buffer[39] = copy.deepcopy(self.episode_buffer[38])[1:]

        node_inputs, node_padding_mask, edge_mask, current_index, current_edge, edge_padding_mask, frontier_distribution, heading_visited, neighbor_best_headings = observation
        self.episode_buffer[11] += node_inputs
        self.episode_buffer[12] += node_padding_mask.bool()
        self.episode_buffer[13] += edge_mask.bool()
        self.episode_buffer[14] += current_index
        self.episode_buffer[15] += current_edge
        self.episode_buffer[16] += edge_padding_mask.bool()
        self.episode_buffer[17] += frontier_distribution
        self.episode_buffer[18] += heading_visited
        self.episode_buffer[39] += neighbor_best_headings

        self.episode_buffer[36] += torch.tensor(next_node_index_list).reshape(1, -1, 1).to(self.device)
        self.episode_buffer[37] = copy.deepcopy(self.episode_buffer[36])[1:]
        self.episode_buffer[37] += copy.deepcopy(self.episode_buffer[36])[-1:]

    def save_ground_truth_observation(self, ground_truth_observation):
        node_inputs, node_padding_mask, edge_mask, current_index, current_edge, edge_padding_mask, frontier_distribution, heading_visited = ground_truth_observation
        self.episode_buffer[19] += node_inputs
        self.episode_buffer[20] += node_padding_mask.bool()
        self.episode_buffer[21] += edge_mask.bool()
        self.episode_buffer[22] += current_index
        self.episode_buffer[23] += current_edge
        self.episode_buffer[24] += edge_padding_mask.bool()
        self.episode_buffer[25] += frontier_distribution
        self.episode_buffer[26] += heading_visited

    def save_next_ground_truth_observations(self, ground_truth_observation):
        self.episode_buffer[27] = copy.deepcopy(self.episode_buffer[19])[1:]
        self.episode_buffer[28] = copy.deepcopy(self.episode_buffer[20])[1:]
        self.episode_buffer[29] = copy.deepcopy(self.episode_buffer[21])[1:]
        self.episode_buffer[30] = copy.deepcopy(self.episode_buffer[22])[1:]
        self.episode_buffer[31] = copy.deepcopy(self.episode_buffer[23])[1:]
        self.episode_buffer[32] = copy.deepcopy(self.episode_buffer[24])[1:]
        self.episode_buffer[33] = copy.deepcopy(self.episode_buffer[25])[1:]
        self.episode_buffer[34] = copy.deepcopy(self.episode_buffer[26])[1:]

        node_inputs, node_padding_mask, edge_mask, current_index, current_edge, edge_padding_mask, frontier_distribution, heading_visited = ground_truth_observation
        self.episode_buffer[27] += node_inputs
        self.episode_buffer[28] += node_padding_mask.bool()
        self.episode_buffer[29] += edge_mask.bool()
        self.episode_buffer[30] += current_index
        self.episode_buffer[31] += current_edge
        self.episode_buffer[32] += edge_padding_mask.bool()
        self.episode_buffer[33] += frontier_distribution
        self.episode_buffer[34] += heading_visited

    def save_all_indices(self, all_agent_indices):
        self.episode_buffer[35] += torch.tensor(all_agent_indices).reshape(1, -1, 1).to(self.device)

