import matplotlib.pyplot as plt
from copy import deepcopy
from matplotlib.patches import Wedge, FancyArrowPatch

from utils.env import Env
from utils.agent import Agent
from utils.utils import *
from utils.node_manager import NodeManager
from utils.ground_truth_node_manager import GroundTruthNodeManager
from utils.model import PolicyNet
from utils.motion_model import compute_allowable_heading  
import time

if not os.path.exists(gifs_path):
    os.makedirs(gifs_path)

class MultiAgentWorker:
    def __init__(self, meta_agent_id, policy_net, global_step, device='cpu', save_image=False):
        self.meta_agent_id = meta_agent_id
        self.global_step = global_step
        self.save_image = save_image
        self.device = device
        self.fov = FOV
        self.sensor_range = SENSOR_RANGE
        self.sim_steps = NUM_SIM_STEPS

        self.env = Env(global_step, self.fov, self.sensor_range, plot=self.save_image)
        self.n_agents = N_AGENTS
        self.node_manager = NodeManager(self.fov, self.sensor_range, plot=self.save_image)
        self.ground_truth_node_manager = GroundTruthNodeManager(self.node_manager, self.env.ground_truth_info, self.sensor_range,
                                                                device=self.device, plot=self.save_image)

        self.robot_list = [Agent(i, policy_net, self.fov, self.env.angles[i], self.sensor_range, self.node_manager, self.ground_truth_node_manager, self.device, self.save_image) for i in
                           range(self.n_agents)]

        self.episode_buffer = []
        self.perf_metrics = dict()
        for i in range(NUM_EPISODE_BUFFER):
            self.episode_buffer.append([])

    def run_episode(self):
        done = False
        for robot in self.robot_list:
            robot.update_graph(self.env.belief_info, self.env.robot_locations[robot.id].copy())
        for robot in self.robot_list:    
            robot.update_planning_state(self.env.robot_locations)

        for i in range(MAX_EPISODE_STEP):

            selected_locations = []
            dist_list = []
            next_node_index_list = []
            next_heading_index_list = []
            for robot in self.robot_list:
                observation = robot.get_observation()
                ground_truth_observation = robot.ground_truth_node_manager.get_ground_truth_observation(robot.location, self.env.robot_locations)

                robot.save_observation(observation)
                robot.save_ground_truth_observation(ground_truth_observation)

                next_location, next_node_index, action_index, next_heading_index = robot.select_next_waypoint(observation)

                robot.save_action(action_index)

                selected_locations.append(next_location)
                dist_list.append(np.linalg.norm(next_location - robot.location))
                next_node_index_list.append(next_node_index)
                next_heading_index_list.append(next_heading_index)

            selected_locations = np.array(selected_locations).reshape(-1, 2)
            arriving_sequence = np.argsort(np.array(dist_list))
            selected_locations_in_arriving_sequence = np.array(selected_locations)[arriving_sequence]

            # Solve collision
            for j, selected_location in enumerate(selected_locations_in_arriving_sequence):
                solved_locations = selected_locations_in_arriving_sequence[:j]
                while selected_location[0] + selected_location[1] * 1j in solved_locations[:, 0] + solved_locations[:, 1] * 1j:
                    id = arriving_sequence[j]
                    nearby_nodes = self.robot_list[id].node_manager.nodes_dict.nearest_neighbors(
                        selected_location.tolist(), 25)
                    for node in nearby_nodes:
                        coords = node.data.coords
                        if coords[0] + coords[1] * 1j in solved_locations[:, 0] + solved_locations[:, 1] * 1j:
                            continue
                        selected_location = coords
                        break

                    selected_locations_in_arriving_sequence[j] = selected_location
                    selected_locations[id] = selected_location


            # Compute simulation data
            robot_locations_sim = []
            robot_headings_sim = []
            all_robots_heading_list = []
            for k, (robot, next_location, next_heading_index) in enumerate(zip(self.robot_list, selected_locations, next_heading_index_list)):
                robot_current_cell = get_cell_position_from_coords(robot.location, self.env.belief_info)
                robot_cell = get_cell_position_from_coords(next_location, self.env.belief_info)

                next_heading = next_heading_index*(360/NUM_ANGLES_BIN)
                final_heading = compute_allowable_heading(robot.location, next_location, robot.heading, next_heading, robot.velocity, robot.yaw_rate)

                # Generate intermediate points
                intermediate_cells = np.linspace(robot_current_cell, robot_cell, self.sim_steps+1)[1:] 

                # Round to nearest integer to get valid cell coordinates
                intermediate_cells = np.round(intermediate_cells).astype(int)
                intermediate_headings = self.smooth_heading_change(robot.heading, final_heading, steps=self.sim_steps)

                robot_locations_sim.append(intermediate_cells)
                robot_headings_sim.append(intermediate_headings)
                all_robots_heading_list.append(final_heading)

                robot.update_heading(final_heading)

            for l in range(self.sim_steps):
                robot_location_sim_step = []
                robot_heading_sim_step = []
                for q in range(self.n_agents):
                    self.env.update_robot_belief(robot_locations_sim[q][l], robot_headings_sim[q][l])
                    robot_location_sim_step.append(robot_locations_sim[q][l])
                    robot_heading_sim_step.append(robot_headings_sim[q][l])
                
                if self.save_image:
                    num_frame = i * self.sim_steps + l
                    self.plot_local_env_sim(num_frame, robot_location_sim_step, robot_heading_sim_step)

            reward_list = []
            for robot, next_location, next_node_index in zip(self.robot_list, selected_locations, next_node_index_list):
                self.env.final_sim_step(next_location, robot.id)

                node = self.node_manager.nodes_dict.find((next_location[0], next_location[1])).data
                observable_frontiers = node.observable_frontiers
                observable_frontiers = np.array(list(observable_frontiers))
                if observable_frontiers.shape[0] > 0:
 
                    coords = np.array(node.coords)

                    delta = observable_frontiers - coords
                    angles = np.degrees(np.arctan2(delta[:, 1], delta[:, 0]) % (2 * np.pi))

                    angle_diff = (angles - robot.heading + 180) % 360 - 180
                    current_observable_frontiers = observable_frontiers[np.abs(angle_diff) <= robot.fov / 2]
 
                    utility_reward = len(current_observable_frontiers) / ((2 * self.sensor_range * 3.14 // FRONTIER_CELL_SIZE) / (360/robot.fov))    

                else:
                    utility_reward = 0

                preferred_angle = node.highest_utility_angle
                if preferred_angle == -360:
                    angle_reward = 0
                else:
                    angle_reward = np.cos(np.radians(robot.heading - preferred_angle))

                trajectory_angle = np.degrees(np.arctan2(next_location[1] - robot.location[1], 
                                               next_location[0] - robot.location[0]) % (2 * np.pi))
                trajectory_reward = np.cos(np.radians(robot.heading - trajectory_angle)) 
                reward_list.append(utility_reward + trajectory_reward)  

                robot.update_graph(self.env.belief_info, self.env.robot_locations[robot.id].copy())

            if self.robot_list[0].utility.sum() == 0:
                done = True

            team_reward = self.env.calculate_team_reward() - 0.5
            if done:
                team_reward += 10

            curr_node_indices = np.array([robot.current_index for robot in self.robot_list])
            for robot, reward in zip(self.robot_list, reward_list):
                robot.save_reward(reward + team_reward)
                robot.save_all_indices(curr_node_indices)
                robot.update_planning_state(self.env.robot_locations)
                robot.save_done(done)

            if done:
                break

        # save metrics
        self.perf_metrics['travel_dist'] = max([robot.travel_dist for robot in self.robot_list])
        self.perf_metrics['explored_rate'] = self.env.explored_rate
        self.perf_metrics['success_rate'] = done

        # save episode buffer
        for robot in self.robot_list:
            observation = robot.get_observation()
            ground_truth_observation = robot.ground_truth_node_manager.get_ground_truth_observation(robot.location, self.env.robot_locations)
            robot.save_next_observations(observation, next_node_index_list)
            robot.save_next_ground_truth_observations(ground_truth_observation)
            for i in range(len(self.episode_buffer)):
                self.episode_buffer[i] += robot.episode_buffer[i]

        # save gif
        if self.save_image:
            make_gif(gifs_path, self.global_step, self.env.frame_files, self.env.explored_rate)

    def smooth_heading_change(self, prev_heading, heading, steps=10):
        # Ensure both angles are in the range [0, 360)
        prev_heading = prev_heading % 360
        heading = heading % 360

        # Calculate the angle difference
        diff = heading - prev_heading
        
        # Adjust for the shortest path
        if abs(diff) > 180:
            diff = diff - 360 if diff > 0 else diff + 360

        # Generate intermediate angles
        intermediate_headings = [
            (prev_heading + i * diff / steps) % 360
            for i in range(1, steps)
        ]

        # Ensure the final heading is exactly the target heading
        intermediate_headings.append(heading)
        return intermediate_headings
            
    def heading_to_vector(self, heading, length=25):
        # Convert heading to vector
        if isinstance(heading, (list, np.ndarray)):
            heading = heading[0]
        heading_rad = np.radians(heading)
        return np.cos(heading_rad) * length, np.sin(heading_rad) * length

    def plot_local_env_sim(self, step, robot_locations, robot_headings):
        plt.switch_backend('agg')
        plt.figure(figsize=(6, 2.5))
        color_list = ['r', 'b', 'g', 'y']
        color_name = ['Red', 'Blue', 'Green', 'Yellow']
        sensing_range = self.sensor_range / CELL_SIZE

        plt.subplot(1, 3, 1)
        plt.imshow(self.env.robot_belief, cmap='gray')
        plt.axis('off')
        xlim = plt.gca().get_xlim()
        ylim = plt.gca().get_ylim()
        plt.xlim(xlim[0], xlim[1])
        plt.ylim(ylim[0], ylim[1])
          
        for robot in self.robot_list:
            c = color_list[robot.id]
            
            if robot.id == 0:
                nodes = get_cell_position_from_coords(robot.node_coords, robot.map_info)
                plt.scatter(nodes[:, 0], nodes[:, 1], c=robot.utility, s=8, zorder=2)
                for i, (x, y) in enumerate(nodes):
                    plt.text(x-3, y-3, f'{robot.utility[i]:.0f}', ha='center', va='bottom', fontsize=3, color='blue', zorder=3)
                   
        # Plot frontiers
        global_frontiers = get_frontier_in_map(self.env.belief_info)
        if len(global_frontiers) != 0:
            frontiers_cell = get_cell_position_from_coords(np.array(list(global_frontiers)), self.env.belief_info) #shape is (2,)
            if len(global_frontiers) == 1:
                frontiers_cell = frontiers_cell.reshape(1,2)
            plt.scatter(frontiers_cell[:, 0], frontiers_cell[:, 1], s=1, c='r')       

        plt.subplot(1, 3, 2)
        plt.imshow(self.env.robot_belief, cmap='gray')
        plt.axis('off')
        plt.xlim(xlim[0], xlim[1])
        plt.ylim(ylim[0], ylim[1])
        color_list = ['r', 'b', 'g', 'y']

        for i, (robot, location, heading) in enumerate(zip(self.robot_list, robot_locations, robot_headings)):
            c = color_list[robot.id]
            dx, dy = self.heading_to_vector(heading, length=sensing_range)
            arrow = FancyArrowPatch((location[0], location[1]), (location[0] + dx/1.25, location[1] + dy/1.25),
                                    mutation_scale=10,
                                    color=c,
                                    arrowstyle='-|>')
            plt.gca().add_artist(arrow)
            plt.text(location[0] + 5, location[1] + 5, f'{heading:.0f}°', color=c, fontsize=6, ha='left', va='center', zorder=16)

            robot_location = get_coords_from_cell_position(location, self.env.belief_info)
            trajectory_x = robot.trajectory_x.copy()
            trajectory_y = robot.trajectory_y.copy()
            trajectory_x.append(robot_location[0])
            trajectory_y.append(robot_location[1])
            plt.plot((np.array(trajectory_x) - robot.map_info.map_origin_x) / robot.cell_size,
                     (np.array(trajectory_y) - robot.map_info.map_origin_y) / robot.cell_size, c,
                     linewidth=1.2, zorder=1)
            
        # Plot frontiers
        if len(global_frontiers) != 0:
            plt.scatter(frontiers_cell[:, 0], frontiers_cell[:, 1], s=1, c='r')

        # Ground truth data
        plt.subplot(1, 3, 3)
        plt.imshow(self.ground_truth_node_manager.ground_truth_map_info.map, cmap='gray')
        plt.xlim(xlim[0], xlim[1])
        plt.ylim(ylim[0], ylim[1])
        for i, (location, heading) in enumerate(zip(robot_locations, robot_headings)):
            c = color_list[i]
            plt.plot(location[0], location[1], c+'o', markersize=6, zorder=5)
            dx, dy = self.heading_to_vector(heading, length=sensing_range)
            plt.arrow(location[0], location[1], dx, dy, head_width=5, head_length=5, fc=c, ec=c, zorder= 15)

            # Draw cone representing field of vision
            cone = Wedge(center=(location[0], location[1]), r=self.sensor_range / CELL_SIZE, theta1=(heading-self.fov/2), 
                         theta2=(heading+self.fov/2), color=c, alpha=0.5, zorder=10)
            plt.gca().add_artist(cone)
            nodes = get_cell_position_from_coords(self.ground_truth_node_manager.ground_truth_node_coords, self.ground_truth_node_manager.ground_truth_map_info)
            plt.scatter(nodes[:, 0], nodes[:, 1], c=self.ground_truth_node_manager.explored_sign, s=8, zorder=2)

        plt.axis('off')
        robot_headings = [f"{color_name[robot.id]}- {robot.heading:.0f}°" for robot in self.robot_list]
        plt.suptitle('Explored ratio: {:.4g}  Travel distance: {:.4g}\nRobot Headings: {}'.format(
            self.env.explored_rate,
            max([robot.travel_dist for robot in self.robot_list]),
            ', '.join(robot_headings)
        ), fontweight='bold', fontsize=10)
        plt.tight_layout()
        plt.savefig('{}/{}_{}_samples.png'.format(gifs_path, self.global_step, step), dpi=150)
        plt.close()
        frame = '{}/{}_{}_samples.png'.format(gifs_path, self.global_step, step)
        self.env.frame_files.append(frame)

if __name__ == '__main__':
    from parameter import *
    import torch
    policy_net = PolicyNet(NODE_INPUT_DIM, EMBEDDING_DIM, NUM_ANGLES_BIN)
    if LOAD_MODEL:
        checkpoint = torch.load(load_path + '/checkpoint.pth', map_location='cpu')
        policy_net.load_state_dict(checkpoint['policy_model'])
        print('Policy loaded!')
    worker = MultiAgentWorker(0, policy_net, 888, 'cpu', True)
    worker.run_episode()
