from importlib import import_module
import numpy as np

class EnvBase:
    """
    Env Base class.
    Env objects are virtual playgrounds for bots to evolve and produce behaviour.
    One environment gets to make interact, at a given time, as many bots as there are populations.
    Subclasses need to be named *Env*.

    :param args: Experiment specific arguments (obtained through argparse).
    :type args: NameSpace
    :param rank: MPI rank of process.
    :type rank: int
    :param size: Number of MPI processes.
    :type size: int
    :param nb_populations: Total number of populations.
    :type nb_populations: int
    :param io_path: Path to this environment's IO class
    :type io_path: string
    """
    def __init__(self, args, rank, size, nb_populations=1, io_path='IO.base'):

        self.args = args
        self.rank = rank
        self.size = size
        self.nb_populations = nb_populations

        self.initialize_io(args, rank, size, io_path)
        self.initialize_bots(args, rank, nb_populations)

    def initialize_io(self, args, rank, size, io_path):
        """
        Called upon object initialization.
        Initializes an IO object.
        IO objects handle file input/output.
        """
        self.io = getattr(import_module(io_path), 'IO')(args, rank, size, self.nb_populations)

    def initialize_bots(self, args, rank, nb_populations):
        """
        Called upon object initialization.
        Initializes bots.
        Bots evolve and produce behaviour in the environment.
        """

        bot_path = args.bots_path.replace('/','.').replace('.py', '')

        self.bots = []

        for pop_nb in range(self.nb_populations):
            self.bots.append( getattr(import_module(bot_path), 'Bot')(args, rank, pop_nb, nb_populations) )
        
    def build_bots(self, seeds):
        """
        Build each bot in the environment from scratch with the full list of seeds.
        """
        for pop_nb in range(self.nb_populations):
            self.bots[pop_nb].build(seeds[pop_nb])

    def extend_bots(self, seeds):
        """
        Extend each bot in the environment with its latest seed.
        """
        for pop_nb in range(self.nb_populations):
            self.bots[pop_nb].extend(seeds[pop_nb])

    def setup_to_run(self):
        """
        Setup each bot in order to then run them within the environment.
        """
        for bot in self.bots:
            bot.setup_to_run()

    def setup_to_save(self):
        """
        Setup bots to then save them (pickled to a file or pickled to be sent to another process).
        """
        for bot in self.bots:
            bot.setup_to_save()

    def evaluate_bots(self, gen_nb):
        """
        Method called once per iteration in order to evaluate and attribute fitnesses to bots.
        The added random jitter to fitnesses is to keep reproducibility accross communication protocols.

        :param gen_nb: Current generation number
        :type gen_nb: int
        """
        self.setup_to_run()

        fitnesses = self.run(gen_nb) + np.random.rand( len(self.bots) ) * 0.0001

        self.setup_to_save()
        
        return np.array(fitnesses, dtype=np.float32)

    def run(self, gen_nb):
        """
        Inner method of *evaluate_bots*.
        Should be implemented and made to return the bots' fitnesses as such :
        % return [bot_0_fitness, bot_1_fitness, ..., bot_n_fitness] %

        :param gen_nb: Current generation number
        :type gen_nb: int
        """
        raise NotImplementedError