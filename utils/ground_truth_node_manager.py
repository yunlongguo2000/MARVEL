"""
Manages ground truth nodes for exploration and mapping in a robotic environment.

This class handles the creation, tracking, and management of nodes representing
the ground truth map, and operations like node generation, utility calculation,
and path planning.

Attributes:
    nodes_dict (QuadTree): A quadtree data structure storing ground truth nodes
    node_manager (object): Manages the current state of nodes
    ground_truth_map_info (object): Contains information about the ground truth map
    sensor_range (float): Maximum sensing range of the robot
    device (str): Computational device for tensor operations
    plot (bool): Flag to enable plotting of ground truth environment

Key Methods:
    initialize_graph(): Creates initial nodes based on ground truth map
    update_graph(): Updates node information from node manager
    get_ground_truth_observation(): Generates observation data for nodes
    Dijkstra(): Computes shortest paths between nodes
    a_star(): Finds optimal path between two nodes
"""
import time
import torch

import numpy as np
from utils.utils import *
from parameter import *
import utils.quads as quads
import matplotlib.pyplot as plt


class GroundTruthNodeManager:
    def __init__(self, node_manager, ground_truth_map_info, sensor_range, device='cpu', plot=False):
        self.nodes_dict = quads.QuadTree((0, 0), 1000, 1000)
        self.node_manager = node_manager
        self.ground_truth_map_info = ground_truth_map_info
        self.ground_truth_node_coords = None
        self.ground_truth_node_utility = None
        self.explored_sign = None
        self.highest_utility_angles = None
        self.frontier_distribution = None
        self.sensor_range = sensor_range
        self.device = device
        self.plot = plot
        self.num_angles_bin = NUM_ANGLES_BIN

        self.initialize_graph()

    def check_node_exist_in_dict(self, coords):
        key = (coords[0], coords[1])
        exist = self.nodes_dict.find(key)
        return exist

    def get_ground_truth_observation(self, robot_location, robot_locations):
        self.update_graph()

        all_node_coords = []
        # Ensure node order is the same as node manager
        for node in self.node_manager.nodes_dict.__iter__():
            all_node_coords.append(node.data.coords)
        for node in self.nodes_dict.__iter__():
            if node.data.explored == 0:
                all_node_coords.append(node.data.coords)
        all_node_coords = np.array(all_node_coords).reshape(-1, 2)
        utility = []
        explored_sign = []
        highest_utility_angles = []
        frontiers_distribution = []
        heading_visited = []

        n_nodes = all_node_coords.shape[0]
        adjacent_matrix = np.ones((n_nodes, n_nodes)).astype(int)
        node_coords_to_check = all_node_coords[:, 0] + all_node_coords[:, 1] * 1j
        for i, coords in enumerate(all_node_coords):
            node = self.nodes_dict.find((coords[0], coords[1])).data
            utility.append(node.utility)
            explored_sign.append(node.explored)
            highest_utility_angles.append(node.highest_utility_angle)
            frontiers_distribution.append(node.frontiers_distribution)
            heading_visited.append(node.heading_visited)
            for neighbor in node.neighbor_list:
                index = np.argwhere(node_coords_to_check == neighbor[0] + neighbor[1] * 1j)
                index = index[0][0]
                adjacent_matrix[i, index] = 0

        utility = np.array(utility)
        explored_sign = np.array(explored_sign)
        highest_utility_angles = np.array(highest_utility_angles)
        frontiers_distribution = np.array(frontiers_distribution)
        heading_visited = np.array(heading_visited)

        current_index = np.argwhere(node_coords_to_check == robot_location[0] + robot_location[1] * 1j)[0][0]
        

        neighbor_indices = []
        current_node_in_belief = self.node_manager.nodes_dict.find(robot_location.tolist()).data
        for neighbor in current_node_in_belief.neighbor_list:
            index = np.argwhere(node_coords_to_check == neighbor[0] + neighbor[1] * 1j)[0][0]
            neighbor_indices.append(index)
        neighbor_indices = np.sort(np.array(neighbor_indices))

        self.ground_truth_node_coords = all_node_coords
        self.ground_truth_node_utility = utility
        self.explored_sign = explored_sign
        self.highest_utility_angles = highest_utility_angles
        self.frontiers_distribution = frontiers_distribution
        self.heading_visited = heading_visited

        indices = np.argwhere(utility > 0).reshape(-1)
        utility_node_coords = all_node_coords[indices]
        dist_dict, prev_dict = self.Dijkstra(robot_location)
        nearest_utility_coords = robot_location
        nearest_dist = 1e8
        for coords in utility_node_coords:
            dist = dist_dict[(coords[0], coords[1])]
            if 0 < dist < nearest_dist:
                nearest_dist = dist
                nearest_utility_coords = coords

        path_coords, dist = self.a_star(robot_location, nearest_utility_coords)
        guidepost = np.zeros_like(utility)
        for coords in path_coords:
            index = np.argwhere(all_node_coords[:, 0] + all_node_coords[:, 1] * 1j == coords[0] + coords[1] * 1j)[0]
            guidepost[index] = 1

        robot_in_graph = self.nodes_dict.nearest_neighbors(robot_location.tolist(), 1)[0].data.coords
        current_index = np.argwhere(node_coords_to_check == robot_in_graph[0] + robot_in_graph[1] * 1j)[0][0]
        neighbor_indices = np.argwhere(adjacent_matrix[current_index] == 0).reshape(-1)

        occupancy = np.zeros((n_nodes, 1))
        for location in robot_locations:
            location_in_graph = self.nodes_dict.find((location[0], location[1])).data.coords
            index = np.argwhere(node_coords_to_check == location_in_graph[0] + location_in_graph[1] * 1j)[0][0]
            if index == current_index:
                occupancy[index] = -1
            else:
                occupancy[index] = 1

        node_coords = all_node_coords
        node_utility = utility.reshape(-1, 1)
        node_explored_sign = explored_sign.reshape(-1, 1)
        node_guidepost = guidepost.reshape(-1, 1)
        node_occupancy = occupancy.reshape(-1, 1)
        node_highest_utility_angles = highest_utility_angles.reshape(-1, 1)
        node_frontiers_distribution = frontiers_distribution.reshape(-1, self.num_angles_bin)
        node_heading_visited = heading_visited.reshape(-1, self.num_angles_bin)
        current_index = current_index
        edge_mask = adjacent_matrix
        current_edge = neighbor_indices
        n_node = node_coords.shape[0]

        current_node_coords = node_coords[current_index]
        node_coords = np.concatenate((node_coords[:, 0].reshape(-1, 1) - current_node_coords[0],
                                      node_coords[:, 1].reshape(-1, 1) - current_node_coords[1]),
                                      axis=-1) / UPDATING_MAP_SIZE / 2

        node_utility = node_utility / (2 * self.sensor_range * 3.14 // FRONTIER_CELL_SIZE)
        node_frontiers_distribution = node_frontiers_distribution / ((2 * self.sensor_range * 3.14 // FRONTIER_CELL_SIZE) / self.num_angles_bin)
        node_highest_utility_angles = node_highest_utility_angles / 360
        node_inputs = np.concatenate((node_coords, node_utility, node_guidepost, node_occupancy, node_highest_utility_angles, node_explored_sign), axis=1)
        node_inputs = torch.FloatTensor(node_inputs).unsqueeze(0).to(self.device)

        assert node_coords.shape[0] < NODE_PADDING_SIZE, print(node_coords.shape[0], NODE_PADDING_SIZE)
        padding = torch.nn.ZeroPad2d((0, 0, 0, NODE_PADDING_SIZE - n_node))
        node_inputs = padding(node_inputs)

        node_frontiers_distribution = torch.Tensor(node_frontiers_distribution).unsqueeze(0).to(self.device)
        node_frontiers_distribution = padding(node_frontiers_distribution)

        node_heading_visited = torch.Tensor(node_heading_visited).unsqueeze(0).to(self.device)
        node_heading_visited = padding(node_heading_visited)

        node_padding_mask = torch.zeros((1, 1, n_node), dtype=torch.int16).to(self.device)
        node_padding = torch.ones((1, 1, NODE_PADDING_SIZE - n_node), dtype=torch.int16).to(
            self.device)
        node_padding_mask = torch.cat((node_padding_mask, node_padding), dim=-1)

        edge_mask = torch.tensor(edge_mask).unsqueeze(0).to(self.device)

        padding = torch.nn.ConstantPad2d(
            (0, NODE_PADDING_SIZE - n_node, 0, NODE_PADDING_SIZE - n_node), 1)
        edge_mask = padding(edge_mask)

        current_in_edge = np.argwhere(current_edge == current_index)[0][0]
        current_edge = torch.tensor(current_edge).unsqueeze(0)
        k_size = current_edge.size()[-1]
        padding = torch.nn.ConstantPad1d((0, K_SIZE - k_size), 0)
        current_edge = padding(current_edge)
        current_edge = current_edge.unsqueeze(-1)

        edge_padding_mask = torch.zeros((1, 1, k_size), dtype=torch.int16).to(self.device)
        edge_padding_mask[0, 0, current_in_edge] = 1
        padding = torch.nn.ConstantPad1d((0, K_SIZE - k_size), 1)
        edge_padding_mask = padding(edge_padding_mask)

        current_index = torch.tensor([current_index]).reshape(1, 1, 1).to(self.device)

        return [node_inputs, node_padding_mask, edge_mask, current_index, current_edge, edge_padding_mask, node_frontiers_distribution, node_heading_visited]
    
    def Dijkstra(self, start):
        q = set()
        dist_dict = {}
        prev_dict = {}

        for node in self.nodes_dict.__iter__():
            coords = node.data.coords
            key = (coords[0], coords[1])
            dist_dict[key] = 1e8
            prev_dict[key] = None
            q.add(key)

        assert (start[0], start[1]) in dist_dict.keys()
        dist_dict[(start[0], start[1])] = 0

        while len(q) > 0:
            u = None
            for coords in q:
                if u is None:
                    u = coords
                elif dist_dict[coords] < dist_dict[u]:
                    u = coords

            q.remove(u)

            if self.nodes_dict.find(u) is None:
                print(u)
                for node in self.nodes_dict.__iter__():
                    print(node.data.coords)

            node = self.nodes_dict.find(u).data
            for neighbor_node_coords in node.neighbor_list:
                v = (neighbor_node_coords[0], neighbor_node_coords[1])
                if v in q:
                    cost = ((neighbor_node_coords[0] - u[0]) ** 2 + (
                            neighbor_node_coords[1] - u[1]) ** 2) ** (1 / 2)
                    cost = np.round(cost, 2)
                    alt = dist_dict[u] + cost
                    if alt < dist_dict[v]:
                        dist_dict[v] = alt
                        prev_dict[v] = u

        return dist_dict, prev_dict
    
    def h(self, coords_1, coords_2):
        h = ((coords_1[0] - coords_2[0]) ** 2 + (coords_1[1] - coords_2[1]) ** 2) ** (1 / 2)
        h = np.round(h, 2)
        return h
    
    def a_star(self, start, destination, boundary=None, max_dist=None):
        if not self.check_node_exist_in_dict(start):
            print(start)
            Warning("start position is not in node dict")
            return [], 1e8
        if not self.check_node_exist_in_dict(destination):
            Warning("end position is not in node dict")
            return [], 1e8

        if start[0] == destination[0] and start[1] == destination[1]:
            return [destination], 0

        open_list = {(start[0], start[1])}
        closed_list = set()
        g = {(start[0], start[1]): 0}
        parents = {(start[0], start[1]): (start[0], start[1])}

        while len(open_list) > 0:
            n = None
            h_n = 1e8

            for v in open_list:
                h_v = self.h(v, destination)
                if n is not None:
                    node = self.nodes_dict.find(n).data
                    n_coords = node.coords
                    h_n = self.h(n_coords, destination)
                if n is None or g[v] + h_v < g[n] + h_n:
                    n = v
                    node = self.nodes_dict.find(n).data
                    n_coords = node.coords

            if max_dist is not None:
                if g[n] > max_dist:
                    return [], 1e8

            if n_coords[0] == destination[0] and n_coords[1] == destination[1]:
                path = []
                length = g[n]
                while parents[n] != n:
                    path.append(n)
                    n = parents[n]
                path.reverse()
                return path, np.round(length, 2)

            for neighbor_node_coords in node.neighbor_list:
                if self.nodes_dict.find(neighbor_node_coords.tolist()) is None:
                    continue
                if boundary is not None:
                    if not (boundary[0] < neighbor_node_coords[0] < boundary[2] and boundary[1] < neighbor_node_coords[1] < boundary[3]):
                        continue
                cost = ((neighbor_node_coords[0] - n_coords[0]) ** 2 + (
                            neighbor_node_coords[1] - n_coords[1]) ** 2) ** (1 / 2)
                cost = np.round(cost, 2)
                m = (neighbor_node_coords[0], neighbor_node_coords[1])
                if m not in open_list and m not in closed_list:
                    open_list.add(m)
                    parents[m] = n
                    g[m] = g[n] + cost
                else:
                    if g[m] > g[n] + cost:
                        g[m] = g[n] + cost
                        parents[m] = n

                        if m in closed_list:
                            closed_list.remove(m)
                            open_list.add(m)
            open_list.remove(n)
            closed_list.add(n)

        print('Path does not exist!')

        return [], 1e8

    def add_node_to_dict(self, coords):
        key = (coords[0], coords[1])
        node = Node(coords, self.sensor_range)
        self.nodes_dict.insert(point=key, data=node)
        return node

    def initialize_graph(self):
        node_coords = self.get_ground_truth_node_coords(self.ground_truth_map_info)
        for coords in node_coords:
            self.add_node_to_dict(coords)

        for node in self.nodes_dict.__iter__():
            node.data.get_neighbor_nodes(self.ground_truth_map_info, self.nodes_dict)
        
    def update_graph(self):
        for node in self.node_manager.nodes_dict.__iter__():
            coords = node.data.coords
            ground_truth_node = self.nodes_dict.find(coords.tolist())
            ground_truth_node.data.utility = node.data.utility
            ground_truth_node.data.explored = 1
            ground_truth_node.data.frontiers_distribution = node.data.frontiers_distribution
            ground_truth_node.data.highest_utility_angle = node.data.highest_utility_angle
            ground_truth_node.data.heading_visited = node.data.heading_visited

    def get_ground_truth_node_coords(location, ground_truth_map_info):
        x_min = ground_truth_map_info.map_origin_x
        y_min = ground_truth_map_info.map_origin_y
        x_max = ground_truth_map_info.map_origin_x + (ground_truth_map_info.map.shape[1] - 1) * CELL_SIZE
        y_max = ground_truth_map_info.map_origin_y + (ground_truth_map_info.map.shape[0] - 1) * CELL_SIZE

        if x_min % NODE_RESOLUTION != 0:
            x_min = (x_min // NODE_RESOLUTION + 1) * NODE_RESOLUTION
        if x_max % NODE_RESOLUTION != 0:
            x_max = x_max // NODE_RESOLUTION * NODE_RESOLUTION
        if y_min % NODE_RESOLUTION != 0:
            y_min = (y_min // NODE_RESOLUTION + 1) * NODE_RESOLUTION
        if y_max % NODE_RESOLUTION != 0:
            y_max = y_max // NODE_RESOLUTION * NODE_RESOLUTION

        x_coords = np.arange(x_min, x_max + 0.1, NODE_RESOLUTION)
        y_coords = np.arange(y_min, y_max + 0.1, NODE_RESOLUTION)
        t1, t2 = np.meshgrid(x_coords, y_coords)
        nodes = np.vstack([t1.T.ravel(), t2.T.ravel()]).T
        nodes = np.around(nodes, 1)

        indices = []
        nodes_cells = get_cell_position_from_coords(nodes, ground_truth_map_info).reshape(-1, 2)
        for i, cell in enumerate(nodes_cells):
            assert 0 <= cell[1] < ground_truth_map_info.map.shape[0] and 0 <= cell[0] < ground_truth_map_info.map.shape[1]
            if ground_truth_map_info.map[cell[1], cell[0]] == FREE:
                indices.append(i)
        indices = np.array(indices)
        nodes = nodes[indices].reshape(-1, 2)

        return nodes

    def plot_ground_truth_env(self, robot_location):
        plt.subplot(1, 3, 3)
        plt.imshow(self.ground_truth_map_info.map, cmap='gray')
        plt.axis('off')
        robot = get_cell_position_from_coords(robot_location, self.ground_truth_map_info)
        nodes = get_cell_position_from_coords(self.ground_truth_node_coords, self.ground_truth_map_info)
        plt.imshow(self.ground_truth_map_info.map, cmap='gray')
        plt.scatter(nodes[:, 0], nodes[:, 1], c=self.explored_sign, zorder=2)
        plt.plot(robot[0], robot[1], 'mo', markersize=16, zorder=5)
        plt.close()


class Node:
    def __init__(self, coords, sensor_range, num_angles_bin=NUM_ANGLES_BIN):
        self.coords = coords
        self.utility = -(sensor_range * 3.14 // FRONTIER_CELL_SIZE)
        self.explored = 0
        self.visited = 0
        self.highest_utility_angle = -360
        self.frontiers_distribution = np.zeros(num_angles_bin)
        self.heading_visited = np.zeros(num_angles_bin)

        self.neighbor_matrix = -np.ones((5, 5))
        self.neighbor_list = []
        self.neighbor_list.append(self.coords)

    def get_neighbor_nodes(self, ground_truth_map_info, nodes_dict):
        center_index = self.neighbor_matrix.shape[0] // 2
        for i in range(self.neighbor_matrix.shape[0]):
            for j in range(self.neighbor_matrix.shape[1]):
                if self.neighbor_matrix[i, j] != -1:
                    continue
                else:
                    if i == center_index and j == center_index:
                        self.neighbor_matrix[i, j] = 1
                        continue

                    neighbor_coords = np.around(np.array([self.coords[0] + (i - center_index) * NODE_RESOLUTION,
                                                self.coords[1] + (j - center_index) * NODE_RESOLUTION]), 1)
                    neighbor_node = nodes_dict.find((neighbor_coords[0], neighbor_coords[1]))
                    if neighbor_node is None:
                        continue
                    else:
                        neighbor_node = neighbor_node.data
                        collision = check_collision(self.coords, neighbor_coords, ground_truth_map_info)
                        neighbor_matrix_x = center_index + (center_index - i)
                        neighbor_matrix_y = center_index + (center_index - j)
                        if not collision:
                            self.neighbor_matrix[i, j] = 1
                            self.neighbor_list.append(neighbor_coords)

                            neighbor_node.neighbor_matrix[neighbor_matrix_x, neighbor_matrix_y] = 1
                            neighbor_node.neighbor_list.append(self.coords)
