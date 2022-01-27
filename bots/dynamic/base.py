import numpy as np

from bots.base import BotBase

class DynamicBotBase(BotBase):
    """
    Dynamic Bot Base class.
    Dynamic bots' nets are of dynamic complexity.
    Subclasses need to be named *Bot*.

    :param pop_nb: Population number.
    :type pop_nb: int
    """
    def __init__(self, args, proc_rank, pop_nb, nb_pops):

        super().__init__(args, proc_rank, pop_nb, nb_pops)

    def initialize_nets(self):
        """
        Initialize the bot's net, should be implemented to create a list containing the networks as such :
        % self.nets = [self.net_0, self.net_1, ..., self.net_n] %
        """
        raise NotImplementedError

    def initialize(self):
        """
        Initialize the bot before it starts getting built.
        """
        self.initialize_nets()

        self.nets_arch_muts = sum([net.architectural_mutations for net in self.nets], [])

        self.nets_archs_initialized = False

    def mutate(self):
        """
        Mutation method for the dynamic bot.
        First initializes the nets' architectures.
        Then, every iteration, mutate network parameters and randomly select N mutations among all net mutations.
        """
        if not self.nets_archs_initialized:

            for net in self.nets:
                net.initialize_architecture()
            
            self.nets_archs_initialized = True

        else:

            for net in self.nets:
                net.mutate_parameters()

            np.random.choice(self.nets_arch_muts)()

    def __call__(self, x):
        """
        Run dynamic bot for one timestep given input 'x'.
        Should be implemented.
        """
        raise NotImplementedError