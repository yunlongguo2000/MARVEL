import matplotlib.pyplot as plt
from copy import deepcopy
from matplotlib.patches import Wedge, FancyArrowPatch
from shapely.geometry import Point, Polygon
from skimage.draw import polygon as sk_polygon

from utils.env_test import Env
from utils.agent import Agent
from utils.utils import *
from utils.node_manager import NodeManager
from utils.ground_truth_node_manager import GroundTruthNodeManager
from utils.model import PolicyNet
from utils.motion_model import compute_allowable_heading  
from test_parameter import *

if not os.path.exists(gifs_path):
    os.makedirs(gifs_path)

class TestWorker:
    def __init__(self, meta_agent_id, policy_net, global_step, n_agent, fov, sensor_range, utility_range, device='cpu', save_image=False, greedy=True):
        self.meta_agent_id = meta_agent_id
        self.global_step = global_step
        self.save_image = save_image
        self.device = device
        self.fov = fov
        self.sim_steps = NUM_SIM_STEPS
        self.sensor_range = sensor_range
        self.greedy = greedy
        self.n_agents = n_agent
        self.scaling = 0.04

        self.env = Env(global_step, self.fov, self.n_agents, self.sensor_range, plot=self.save_image)
        self.node_manager = NodeManager(self.fov, self.sensor_range, utility_range, plot=self.save_image)
        self.ground_truth_node_manager = GroundTruthNodeManager(self.node_manager, self.env.ground_truth_info, self.sensor_range,
                                                                device=self.device, plot=self.save_image)
        self.robot_list = [Agent(i, policy_net, self.fov, self.env.angles[i], self.sensor_range, self.node_manager, None, self.device, self.save_image) for i in
                           range(self.n_agents)]

        self.perf_metrics = dict()

    def run_episode(self):
        done = False
        for robot in self.robot_list:
            robot.update_graph(self.env.belief_info, self.env.robot_locations[robot.id].copy())
        for robot in self.robot_list:    
            robot.update_planning_state(self.env.robot_locations)
        
        reach_checkpoint = False

        max_travel_dist = 0
        trajectory_length = 0

        length_history = [max_travel_dist]
        explored_rate_history = [self.env.explored_rate]
        overlap_rate = self.compute_overlap_rate(self.env.robot_locations, self.env.angles)
        overlap_ratio_history = [overlap_rate]

        setpoints = [[] for _ in range(self.n_agents)]
        headings = [[] for _ in range(self.n_agents)]


        for i in range(MAX_EPISODE_STEP):
            # print(' Current timestep: {}/{}'.format(i, MAX_EPISODE_STEP))
            selected_locations = []
            dist_list = []
            next_node_index_list = []
            next_heading_index_list = []
            for robot in self.robot_list:
                observation = robot.get_observation(pad=False)
              
                next_location, next_node_index, _, next_heading_index = robot.select_next_waypoint(observation, greedy=self.greedy)

                selected_locations.append(next_location)
                dist_list.append(np.linalg.norm(next_location - robot.location))
                next_node_index_list.append(next_node_index)
                next_heading_index_list.append(next_heading_index)

            selected_locations = np.array(selected_locations).reshape(-1, 2)
            arriving_sequence = np.argsort(np.array(dist_list))
            selected_locations_in_arriving_sequence = np.array(selected_locations)[arriving_sequence]

  
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

            for p, location in enumerate(selected_locations):
                setpoints[p].append(location*self.scaling)

            # Compute simulation data
            robot_locations_sim = []
            robot_headings_sim = []
            all_robots_heading_list = []
            for k, (robot, next_location, next_heading_index) in enumerate(zip(self.robot_list, selected_locations, next_heading_index_list)):
                robot_current_cell = get_cell_position_from_coords(robot.location, self.env.belief_info)
                robot_cell = get_cell_position_from_coords(next_location, self.env.belief_info)

                next_heading = next_heading_index*(360/NUM_ANGLES_BIN)
                final_heading = compute_allowable_heading(robot.location, next_location, robot.heading, next_heading, robot.velocity, robot.yaw_rate)

                intermediate_cells = np.linspace(robot_current_cell, robot_cell, self.sim_steps+1)[1:] 

                intermediate_cells = np.round(intermediate_cells).astype(int)
                intermediate_headings = self.smooth_heading_change(robot.heading, final_heading, steps=self.sim_steps)

                robot_locations_sim.append(intermediate_cells)
                robot_headings_sim.append(intermediate_headings)
                all_robots_heading_list.append(final_heading)
                corrected_heading = self.correct_heading(final_heading)
                headings[k].append(corrected_heading)

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

            for robot, next_location, next_node_index in zip(self.robot_list, selected_locations, next_node_index_list):
                self.env.final_sim_step(next_location, robot.id)

                robot.update_graph(self.env.belief_info, self.env.robot_locations[robot.id].copy())

            overlap_rate = self.compute_overlap_rate(selected_locations, all_robots_heading_list)

            for robot in self.robot_list:
                robot.update_planning_state(self.env.robot_locations)

            max_travel_dist += np.max(dist_list)
            length_history.append(max_travel_dist)
            explored_rate_history.append(self.env.explored_rate)
            overlap_ratio_history.append(overlap_rate)
            if self.env.explored_rate > INITIAL_EXPLORED_RATE and not reach_checkpoint:
                trajectory_length = max([robot.travel_dist for robot in self.robot_list])
                reach_checkpoint = True

            if self.env.explored_rate > 0.99:
                done = True

            if done:
                break

        # Save metrics
        self.perf_metrics['travel_dist'] = max([robot.travel_dist for robot in self.robot_list])
        self.perf_metrics['explored_rate'] = self.env.explored_rate
        self.perf_metrics['success_rate'] = done
        if trajectory_length > 0:
            self.perf_metrics['dist_to_0_90'] = trajectory_length
        else:
            self.perf_metrics['dist_to_0_90'] = []
        self.perf_metrics['length_history'] = length_history
        self.perf_metrics['explored_rate_history'] = explored_rate_history
        self.perf_metrics['overlap_ratio_history'] = overlap_ratio_history
    
        # Save gif
        if self.save_image:
            pass
            make_gif_test(gifs_path, self.global_step, self.env.frame_files, self.env.explored_rate, self.n_agents, self.fov, self.sensor_range)

    def smooth_heading_change(self, prev_heading, heading, steps=10):
        prev_heading = prev_heading % 360
        heading = heading % 360
        diff = heading - prev_heading
        
        if abs(diff) > 180:
            diff = diff - 360 if diff > 0 else diff + 360

        intermediate_headings = [
            (prev_heading + i * diff / steps) % 360
            for i in range(1, steps)
        ]

        intermediate_headings.append(heading)
        return intermediate_headings
            
    def heading_to_vector(self, heading, length=25):
        # Convert heading to vector
        if isinstance(heading, (list, np.ndarray)):
            heading = heading[0]
        heading_rad = np.radians(heading)
        return np.cos(heading_rad) * length, np.sin(heading_rad) * length
    
    def create_sensing_mask(self, location, heading):
        mask = np.zeros_like(self.env.ground_truth)

        location_cell = get_cell_position_from_coords(location, self.env.belief_info)
        robot_point = Point(location_cell)

        start_angle = (heading - self.fov / 2 + 360) % 360
        end_angle = (heading + self.fov / 2) % 360

        sector_points = [robot_point]
        if start_angle <= end_angle:
            angle_range = np.linspace(start_angle, end_angle, 20)
        else:
            angle_range = np.concatenate([np.linspace(start_angle, 360, 10), np.linspace(0, end_angle, 10)])
        for angle in angle_range:  
            x = location_cell[0] + SENSOR_RANGE/CELL_SIZE * np.cos(np.radians(angle))
            y = location_cell[1] + SENSOR_RANGE/CELL_SIZE * np.sin(np.radians(angle))
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
        
        free_connected_map = get_free_and_connected_map(location, self.env.belief_info)

        mask[rr, cc] = (free_connected_map[rr, cc] == free_connected_map[location_cell[1], location_cell[0]])
       
        return mask
    
    def compute_overlap_rate(self, all_robots_locations, robot_headings_list):
        all_robot_sensing_mask = []
        for robot_location, robot_heading in zip(all_robots_locations, robot_headings_list):
            robot_sensing_mask = self.create_sensing_mask(robot_location, robot_heading)
            all_robot_sensing_mask.append(robot_sensing_mask)
        
        total_mask = np.sum(all_robot_sensing_mask, axis=0)
        total_sensing_area = np.sum(total_mask > 0)
        total_overlap_area = np.sum(total_mask > 1)

        overlap_ratio = total_overlap_area / total_sensing_area  
        
        return overlap_ratio
    def plot_local_env_sim(self, step, robot_locations, robot_headings):
        plt.switch_backend('agg')
        plt.figure(figsize=(6, 3))
        color_list = ['r', 'b', 'g', 'y']
        color_name = ['Red', 'Blue', 'Green', 'Yellow']
        sensing_range = SENSOR_RANGE / CELL_SIZE

        plt.subplot(1, 2, 1)
        plt.imshow(self.env.robot_belief, cmap='gray')
        plt.axis('off')
        xlim = plt.gca().get_xlim()
        ylim = plt.gca().get_ylim()
        plt.xlim(xlim[0], xlim[1])
        plt.ylim(ylim[0], ylim[1])

        for i, (robot, location, heading) in enumerate(zip(self.robot_list, robot_locations,robot_headings)):
            plot_id = robot.id % 4
            c = color_list[plot_id]
            dx, dy = self.heading_to_vector(heading, length=sensing_range)
            arrow = FancyArrowPatch((location[0], location[1]), (location[0] + dx/1.25, location[1] + dy/1.25),
                                    mutation_scale=10,
                                    color=c,
                                    arrowstyle='-|>')
            plt.gca().add_artist(arrow)
            robot_location = get_coords_from_cell_position(location, self.env.belief_info)
            trajectory_x = robot.trajectory_x.copy()
            trajectory_y = robot.trajectory_y.copy()
            trajectory_x.append(robot_location[0])
            trajectory_y.append(robot_location[1])
            plt.plot((np.array(trajectory_x) - robot.map_info.map_origin_x) / robot.cell_size,
                     (np.array(trajectory_y) - robot.map_info.map_origin_y) / robot.cell_size, c,
                     linewidth=1.2, zorder=1)

        global_frontiers = get_frontier_in_map(self.env.belief_info)
        if len(global_frontiers) != 0:
            frontiers_cell = get_cell_position_from_coords(np.array(list(global_frontiers)), self.env.belief_info) #shape is (2,)
            if len(global_frontiers) == 1:
                frontiers_cell = frontiers_cell.reshape(1,2)
            plt.scatter(frontiers_cell[:, 0], frontiers_cell[:, 1], s=1, c='r')       

        plt.subplot(1, 2, 2)
        plt.imshow(self.env.robot_belief, cmap='gray')
        plt.axis('off')
        plt.xlim(xlim[0], xlim[1])
        plt.ylim(ylim[0], ylim[1])

        for i, (robot, location, heading) in enumerate(zip(self.robot_list, robot_locations,robot_headings)):
            plot_id = robot.id % 4
            c = color_list[plot_id]
            dx, dy = self.heading_to_vector(heading, length=sensing_range)
            arrow = FancyArrowPatch((location[0], location[1]), (location[0] + dx/1.25, location[1] + dy/1.25),
                                    mutation_scale=10,
                                    color=c,
                                    arrowstyle='-|>')
            plt.gca().add_artist(arrow)
           
            # Draw cone representing field of vision
            cone = Wedge(center=(location[0], location[1]), r=SENSOR_RANGE / CELL_SIZE, theta1=(heading-self.fov/2), 
                         theta2=(heading+self.fov/2), color=c, alpha=0.5, zorder=10)
            plt.gca().add_artist(cone)

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
            plt.scatter(frontiers_cell[:, 0], frontiers_cell[:, 1], s=3, c='r')

        plt.axis('off')
        robot_headings = [f"{color_name[robot.id%4]}- {robot.heading:.0f}°" for robot in self.robot_list]
        plt.suptitle('Explored ratio: {:.4g}  Travel distance: {:.4g}\nRobot Headings: {}'.format(
            self.env.explored_rate,
            max([robot.travel_dist for robot in self.robot_list]),
            ', '.join(robot_headings)
        ), fontweight='bold', fontsize=10)
        plt.tight_layout()
        plt.savefig('{}/{}_{}_{}_{}_{}_samples.png'.format(gifs_path, self.global_step, step, self.n_agents, self.fov, self.sensor_range), dpi=150)
        plt.close()
        frame = '{}/{}_{}_{}_{}_{}_samples.png'.format(gifs_path, self.global_step, step, self.n_agents, self.fov, self.sensor_range)
        self.env.frame_files.append(frame)

    def correct_heading(self, heading):
        heading = abs(((heading + 90) % 360) - 360)
        return heading

if __name__ == '__main__':
    import torch
    policy_net = PolicyNet(NODE_INPUT_DIM, EMBEDDING_DIM, NUM_ANGLES_BIN)
    if LOAD_MODEL:
        checkpoint = torch.load(load_path + '/checkpoint.pth', map_location='cpu')
        policy_net.load_state_dict(checkpoint['policy_model'])
        print('Policy loaded!')
    worker = TestWorker(0, policy_net, 188, 4, 120, 10, 'cpu', True)
    worker.run_episode()
