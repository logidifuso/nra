import torch

from bots.base import BotBase

class StaticBotBase(BotBase):
    """
    Static Bot Base class.
    Static Bot objects contain static-sized PyTorch Neural Networks.
    Subclasses need to be named *Bot*.
    
    :param pop_nb: Population number.
    :type pop_nb: int
    """
    def __init__(self, args, rank, pop_nb, nb_pops):

        super().__init__(args, rank, pop_nb, nb_pops)

    def initialize(self):
        """
        Initialize the bot before it starts getting built.
        """
        if 'transfer' in self.args.additional_arguments and self.args.additional_arguments['transfer'] == 1:
            
            self.state = None
            self.obs = None
            self.done = True
            self.nb_elapsed_obs = 0

        self.initialize_nets()

        self.setup_device_and_parameters()

    def initialize_nets(self):
        """
        Initialize the bot's net, should be implemented to create a list containing the networks as such :
        % self.nets = [self.net_0, self.net_1, ..., self.net_n] %
        """
        raise NotImplementedError

    def setup_device_and_parameters(self):
        """
        Links up the bot's net to either CPU or GPU.
        If GPUs are available, nets are equally distributed accross GPUs, otherwise they are set up on CPUs.
        """
        if self.args.enable_gpu_use and torch.cuda.device_count() > 0:
            self.device = 'cuda:' + str( self.rank % torch.cuda.device_count() )
        else:
            self.device = 'cpu'

        for net in self.nets:

            net.device = self.device

            for parameter in net.parameters():

                parameter.requires_grad = False
                parameter.data = torch.zeros_like(parameter.data)

    def mutate(self):
        """
        Mutation method for the static bot.
        Mutates the networks' parameters.
        """
        for net in self.nets:

            for parameter in net.parameters():
                parameter.data += 0.01 * torch.randn_like(parameter.data)

    def __call__(self, x):
        """
        Run static bot for one timestep given input 'x'.
        Should be implemented.
        """
        raise NotImplementedError