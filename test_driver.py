"""
Runs distributed testing for a multi-agent exploration policy using Ray.

This function initializes a distributed testing framework where multiple meta-agents 
run test episodes with varying parameters such as number of agents, field of view, 
and sensor range. It collects and aggregates performance metrics across different 
test configurations.

Key operations:
- Loads a pre-trained policy network
- Distributes test jobs across multiple Ray workers
- Runs tests with different experimental parameters
- Collects and prints performance metrics including:
  - Travel distance
  - Exploration rate
  - Success rate
  - Overlap ratio

The function supports GPU acceleration and allows configurable testing parameters.
"""
import ray
import numpy as np
import torch
import os
import time

from utils.model import PolicyNet
from utils.test_worker import TestWorker
from test_parameter import *
import csv

def run_test():
    device = torch.device('cuda') if USE_GPU else torch.device('cpu')
    global_network = PolicyNet(NODE_INPUT_DIM, EMBEDDING_DIM, NUM_ANGLES_BIN).to(device)

    if device == 'cuda':
        checkpoint = torch.load(f'{load_path}/checkpoint.pth', map_location=torch.device('cuda'))
    else:
        checkpoint = torch.load(f'{load_path}/checkpoint.pth', map_location=torch.device('cpu'))

    global_network.load_state_dict(checkpoint['policy_model'])
    
    meta_agents = [Runner.remote(i) for i in range(NUM_META_AGENT)]
    weights = global_network.state_dict()

    all_fov = [120]
    all_n_agent = [6]
    all_sensor_range = [10]
    all_utility_range = [range_val * 0.9 for range_val in all_sensor_range]

    for n_agent in all_n_agent:
        for fov in all_fov:
            for sensor_range, utility_range in zip(all_sensor_range, all_utility_range):

                curr_test = 0

                dist_history = []
                explore_rate = []
                success_rate = []
                dist_to_0_90 = []
                all_length_history = []
                all_explored_rate_history = []
                all_overlap_ratio_history =[]

                job_list = []
                for i, meta_agent in enumerate(meta_agents):
                    job_list.append(meta_agent.job.remote(weights, curr_test, n_agent, fov, sensor_range, utility_range))
                    curr_test += 1

                try:
                    while len(dist_history) < curr_test:
                        done_id, job_list = ray.wait(job_list)
                        done_jobs = ray.get(done_id)

                        for job in done_jobs:
                            metrics, info = job
                            dist_history.append(metrics['travel_dist'])
                            explore_rate.append(metrics['explored_rate'])
                            success_rate.append(metrics['success_rate'])
                            if metrics['dist_to_0_90']:
                                dist_to_0_90.append(metrics['dist_to_0_90'])
                            all_length_history.extend(metrics['length_history'])
                            all_explored_rate_history.extend(metrics['explored_rate_history'])
                            all_overlap_ratio_history.extend(metrics['overlap_ratio_history'])

                            if curr_test < NUM_TEST:
                                job_list.append(meta_agents[info['id']].job.remote(weights, curr_test, n_agent, fov, sensor_range, utility_range))
                                curr_test += 1

                    print('|#Test set:', TEST_SET)
                    print('|#Total test:', NUM_TEST)
                    print('|#Number of agents:', n_agent)
                    print('|#FOV (degrees):', fov)
                    print('|#Sensor range (m):', sensor_range)
                    print('|#Average max length:', np.array(dist_history).mean())
                    print('|#Max max length:', np.array(dist_history).max())
                    print('|#Min max length:', np.array(dist_history).min())
                    print('|#Std max length:', np.array(dist_history).std())
                    print('|#Average explored rate:', np.array(explore_rate).mean())
                    print('|#Average success rate:', np.array(success_rate).mean())
                    print('|#Average distance to {} explored:'.format(INITIAL_EXPLORED_RATE), np.array(dist_to_0_90).mean())
                    print('|#Std distance to {} explored:'.format(INITIAL_EXPLORED_RATE), np.array(dist_to_0_90).std())
                    print('|#Average overlap ratio:', np.array(all_overlap_ratio_history).mean())
                    print('|#Std overlap ratio:', np.array(all_overlap_ratio_history).std())

                except KeyboardInterrupt:
                    print("CTRL_C pressed. Killing remote workers")
                    for a in meta_agents:
                        ray.kill(a)


@ray.remote(num_cpus=1, num_gpus=NUM_GPU/NUM_META_AGENT)
class Runner(object):
    def __init__(self, meta_agent_id):
        self.meta_agent_id = meta_agent_id
        self.device = torch.device('cuda') if USE_GPU else torch.device('cpu')
        self.local_network = PolicyNet(NODE_INPUT_DIM, EMBEDDING_DIM, NUM_ANGLES_BIN)
        self.local_network.to(self.device)

    def set_weights(self, weights):
        self.local_network.load_state_dict(weights)

    def do_job(self, episode_number, n_agent, fov, sensor_range, utility_range):
        if SAVE_GIFS:
            save_img = True if episode_number % SAVE_IMG_GAP == 0 else False
        else:
            save_img = False
        worker = TestWorker(self.meta_agent_id, self.local_network, episode_number, n_agent, fov, sensor_range, utility_range, device=self.device, save_image=save_img, greedy=GREEDY)
        worker.run_episode()

        perf_metrics = worker.perf_metrics
        return perf_metrics

    def job(self, weights, episode_number, n_agent, fov, sensor_range, utility_range):
        print("Starting episode {} on metaAgent {}".format(episode_number, self.meta_agent_id))
        # set the local weights to the global weight values from the master network
        self.set_weights(weights)

        metrics = self.do_job(episode_number,  n_agent, fov, sensor_range, utility_range)

        info = {
            "id": self.meta_agent_id,
            "episode_number": episode_number,
        }

        return metrics, info


if __name__ == '__main__':
    start_time = time.time()
    ray.init()
    for i in range(NUM_RUN):
        run_test()
    print('Total time taken: {}'.format(time.time() - start_time))
