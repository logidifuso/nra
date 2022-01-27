import numpy as np
import torch
import random

class BotBase:
    """
    Bot Base class.
    Bot objects are artificial agents to be ran and evaluated in environments.
    Bot objects contain Net (Artificial Neural Networks) objects.
    Subclasses need to be named *Bot*.
    """
    def __init__(self, args, rank, pop_nb, nb_pops):

        self.args = args
        self.rank = rank
        self.pop_nb = pop_nb
        self.nb_pops = nb_pops

    def initialize_nets(self):
        """
        Initialize the bot's nets, should be implemented to create a list containing the networks as such :
        % self.nets = [self.net_0, self.net_1, ..., self.net_n] %
        """
        raise NotImplementedError

    def initialize(self):
        """
        Initialize the bot before it starts getting built.
        """
        pass

    def build(self, seeds):
        """
        Build the bot from scratch (N mutations) using its full list of seeds.
        This method is only called on generation 1 for 'ps_p2p' and 'big_ps_p2p'
        but called every generation for the 'ps' protocol.
        """
        self.initialize()

        for seed in seeds:
            self.extend(seed)

    def extend(self, seed):
        """
        Extend the bot (1 mutation) based on a newly generated seed.
        Called every generation for the 'ps_p2p' and 'big_ps_p2p' protocols
        """
        if seed > 0:
            
            np.random.seed(seed)
            torch.manual_seed(seed)
            random.seed(seed)
            
            self.mutate()

    def mutate(self):
        """
        Method to mutate the bot.
        Should be implemented.
        """
        raise NotImplementedError

    def setup_to_run(self):
        """
        Setup the bot and its nets to then be run in an environment.
        """
        self.pre_setup_to_run()

        for net in self.nets:
            net.setup_to_run()

    def pre_setup_to_run(self):
        """
        Placeholder method that can be used for user-defined pre-run setup.
        Can be implemented.
        """
        pass
        
    def setup_to_save(self):
        """
        Setup the bot and its nets to then be pickled (to a file or to be sent to another process).
        """
        self.pre_setup_to_save()

        for net in self.nets:
            net.setup_to_save()

    def pre_setup_to_save(self):
        """
        Placeholder method that can be used for user-defined pre-run setup.
        Can be implemented.
        """
        pass

    def reset(self):
        """
        Reset the bot and its nets' inner states.
        """
        for net in self.nets:
            net.reset()

    def __call__(self, x):
        """
        Run bot for one timestep given input 'x'.
        Should be implemented.
        """
        raise NotImplementedError