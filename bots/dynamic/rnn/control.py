import numpy as np

from bots.dynamic.base import DynamicBotBase
from nets.dynamic.recurrent import Net
from utils.functions.gym import get_info

class Bot(DynamicBotBase):

    def __init__(self, args, rank, pop_nb, nb_pops):

        super().__init__(args, rank, pop_nb, nb_pops)

    def initialize_nets(self):

        d_input, d_output, self.discrete_output, self.output_range = get_info(self.args.additional_arguments['task'])

        self.net = Net(d_input, d_output)

        self.nets = [self.net]

        self.mean = self.v = self.std = self.n = 0

    def update_mean_std(self, x):

        temp_m = self.mean + (x - self.mean) / self.n
        temp_v = self.v + (x - self.mean) * (x - temp_m)

        self.v = temp_v
        self.mean = temp_m
        self.std = np.sqrt(self.v / self.n)

    def env_to_net(self, x):
        
        if hasattr(self, 'n'): # Backward compatibility
            self.n += 1

        non_standardized_tasks = ['acrobot', 'cart_pole', 'mountain_car','mountain_car_continuous',
                                  'bipedal_walker', 'bipedal_walker_hardcore', 'lunar_lander',
                                  'lunar_lander_continuous', 'humanoid_standup',
                                  'inverted_double_pendulum', 'inverted_pendulum', 'swimmer']

        if self.args.additional_arguments['task'] in non_standardized_tasks: 
            return x # Listed tasks did not implement a running standardization in the reported experiments

        self.update_mean_std(x)
        x = (x - self.mean) / (self.std + (self.std == 0))

        return x

    def net_to_env(self, x):
        
        x = np.array(x).squeeze(axis=1)

        if self.discrete_output:
            x = np.argmax(x)
        else:
            x = np.minimum(x, 1) * (self.output_range * 2) - self.output_range
            
        return x
    
    def __call__(self, x):
    
        x = self.env_to_net(x)
        x = self.net(x)
        x = self.net_to_env(x)

        return x