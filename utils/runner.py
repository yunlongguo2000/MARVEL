"""
A runner class for managing multi-agent reinforcement learning episodes using Ray distributed computing.

This class handles the initialization, weight management, and episode execution for a meta-agent 
in a distributed reinforcement learning setup. It supports GPU and CPU configurations and 
interfaces with a policy network and multi-agent worker.

Attributes:
    meta_agent_id (int): Unique identifier for the meta-agent.
    device (torch.device): Computational device (GPU or CPU) for running the network.
    network (PolicyNet): Neural network for policy learning.

Methods:
    get_weights(): Retrieves the current policy network weights.
    set_policy_net_weights(weights): Updates the policy network with given weights.
    do_job(episode_number): Runs a single episode for the meta-agent.
    job(weights_set, episode_number): Executes an episode with specified weights.
"""
import torch
import ray
from utils.model import PolicyNet
from utils.multi_agent_worker import MultiAgentWorker
from parameter import *


class Runner(object):
    def __init__(self, meta_agent_id):
        self.meta_agent_id = meta_agent_id
        self.device = torch.device('cuda') if USE_GPU else torch.device('cpu')
        self.network = PolicyNet(NODE_INPUT_DIM, EMBEDDING_DIM, NUM_ANGLES_BIN)
        self.network.to(self.device)

    def get_weights(self):
        return self.network.state_dict()

    def set_policy_net_weights(self, weights):
        self.network.load_state_dict(weights)

    def do_job(self, episode_number):
        save_img = True if episode_number % SAVE_IMG_GAP == 0 else False
        worker = MultiAgentWorker(self.meta_agent_id, self.network, episode_number, device=self.device, save_image=save_img)
        worker.run_episode()

        job_results = worker.episode_buffer
        perf_metrics = worker.perf_metrics
        return job_results, perf_metrics

    def job(self, weights_set, episode_number):
        print("Starting episode {} on metaAgent {}".format(episode_number, self.meta_agent_id))
        self.set_policy_net_weights(weights_set[0])

        job_results, metrics = self.do_job(episode_number)

        info = {"id": self.meta_agent_id, "episode_number": episode_number}

        return job_results, metrics, info


if USE_GPU:
    gpu = NUM_GPU / NUM_META_AGENT
else:
    gpu = 0

@ray.remote(num_cpus=1, num_gpus=gpu)
class RLRunner(Runner):
    def __init__(self, meta_agent_id):
        super().__init__(meta_agent_id)


if __name__ == '__main__':
    ray.init()
    runner = RLRunner.remote(0)
    job_id = runner.do_job.remote(47)
    out = ray.get(job_id)
    print(out[1])
