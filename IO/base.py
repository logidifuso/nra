import glob
import numpy as np
import os
import pickle

class IOBase:
    """
    IO Base class.
    IO objects are used to handle input/output files.
    Subclasses need to be named *IO* (a default subclass is defined at the bottom of this file).

    :param args: Experiment specific arguments (obtained through argparse).
    :type args: Namespace
    :param rank: MPI rank of process.
    :type rank: int
    :param size: Number of MPI processes.
    :type size: int
    :param nb_populations: Number of different populations represented in the environment.
    :type nb_populations: int
    """
    def __init__(self, args, rank, size, nb_populations):

        self.args = args
        self.rank = rank
        self.size = size
        self.nb_populations = nb_populations

        if self.args.population_size % 2 != 0:
            raise RuntimeError("`args.population_size` must be an even number.")

        if self.args.population_size % self.size != 0:
            raise RuntimeError("`args.population_size` must be a multiple of number of MPI processes.")

        self.setup_elitism()
        self.setup_save_points()
        self.setup_state_path()

    def setup_elitism(self):
        """
        Method called upon object initialization.
        Sets up the 'elitism' parameter which manages how many of the
        best performing bots will not be mutated every iteration.
        """
        if self.args.elitism < 0:

            raise RuntimeError("`args.elitism` must not be < 0.")

        if self.args.elitism < 1:

            if self.args.elitism > 0.5:
                raise RuntimeError("`args.elitism` must not be in ]0.5,1[.")

            self.args.elitism = int(self.args.elitism * self.args.population_size)

        else:

            if self.args.elitism > self.args.population_size // 2:
                raise RuntimeError("`args.elitism` must be < 'population_size'/2.")

            self.args.elitism = int(self.args.elitism)

    def setup_save_points(self):
        """
        Method called upon object initialization.
        Sets up points at which to save the experiment's current state.
        """
        if self.args.save_frequency > self.args.nb_generations or self.args.save_frequency < 0:
            raise RuntimeError("`args.save_frequency` must be in [0, nb_generations].")

        self.save_points = [self.args.nb_elapsed_generations + self.args.nb_generations]
        
        if self.args.save_frequency == 0:
            return

        for i in range(self.args.nb_generations//self.args.save_frequency):
            self.save_points.append( self.args.nb_elapsed_generations + self.args.save_frequency * (i+1) )

    def setup_state_path(self):
        """
        Method called upon object initialization.
        Sets up the path which states will be loaded from and saved into.
        """
        self.path = 'data/states/' + self.args.env_path.replace('/', '.').replace('.py', '') + '/'

        if self.args.additional_arguments == {}:

            self.path += '~'

        else:

            for key in sorted(self.args.additional_arguments):
                self.path += str(key) + '.' + str(self.args.additional_arguments[key]) + '~'

            self.path = self.path[:-1] + '/'

        self.path += self.args.bots_path.replace('/', '.').replace('.py', '') + '/'

    def generate_new_seeds(self, gen_nb):
        """
        Method that produces new seeds meant to mutate the bots for this generation.
        (seed = 0 <=> no mutation)

        :param gen_nb: Current generation number
        :type args: int
        """
        d_0_non_zero = self.args.population_size if gen_nb == 0 else self.args.population_size - self.args.elitism
        d_0_zero = 0 if gen_nb == 0 else self.args.elitism

        non_zero_seeds = np.random.randint(1, 2**32, (d_0_non_zero, 1, 1), dtype=np.uint32)
        zero_seeds = np.zeros((d_0_zero, 1, 1), dtype=np.uint32)
        new_seeds = np.concatenate((non_zero_seeds, zero_seeds), axis=0)

        return new_seeds

    def load_state(self):
        """
        Load a previous experiment's state.
        """
        load_path = self.path + str(self.args.population_size) + '/' + str(self.args.nb_elapsed_generations) + '/'

        pkl_files = [os.path.basename(x) for x in glob.glob(load_path + '*.pkl')]

        if 'net.pkl' in pkl_files:
            pkl_files.remove('net.pkl')

        if not os.path.isdir(load_path) or len(pkl_files) == 0:
            raise RuntimeError("No saved state found at " + load_path + ".")

        if (self.args.communication == 'ps' or self.args.communication == 'ps_p2p') and len(pkl_files) > 1:
            raise RuntimeError("`args.communication` = '" + self.args.communication + "' \
                                while the saved state used 'big_ps_p2p'")

        if self.args.communication == 'big_ps_p2p' and len(pkl_files) != self.size:
            raise RuntimeError("The current number of MPI processes is " + str(self.size) + " \
                               while the saved state made use of " + len(pkl_files) + ".")
        
        if not os.path.isfile(load_path + str(self.rank) + '.pkl'):
            raise RuntimeError("File " + str(self.rank) + ".pkl missing. Unable to load save state.")

        with open(load_path + str(self.rank) + '.pkl', 'rb') as f:
            state = pickle.load(f)

        if self.args.communication == 'ps' and len(state) == 4:
            raise RuntimeError("`args.communication` = 'ps' while the saved state used 'ps_p2p'")

        if self.args.communication == 'ps_p2p' and len(state) == 3:
            raise RuntimeError("`args.communication` = 'ps_p2p' while the saved state used 'ps'")

        return state

    def save_state(self, state, gen_nb):
        """
        Save the current experiment's state.

        :param gen_nb: Current generation number
        :type args: int
        """
        save_path = self.path + str(self.args.population_size) + '/' + str(gen_nb) + '/'

        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        
        with open(save_path + str(self.rank) + '.pkl', 'wb') as f:
            pickle.dump(state, f)

class IO(IOBase):
    pass