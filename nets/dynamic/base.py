class DynamicNetBase:
    """
    Dynamic Net Base class.
    Dynamic Net objects are Artificial Neural Networks of Dynamic Complexity.
    Subclasses need to be named *Net*.
    """
    def __init__(self):
        """
        Should be implemented to create a list containing the net's architectural mutations as such :
        % self.architectural_mutations = [self.mutation_0, self.mutation_1, ..., self.mutation__n] %
        Every iteration, some of these mutations will be randomly sampled and executed.
        """
        raise NotImplementedError

    def initialize_architecture(self):
        """
        Initialize the net's architecture to set it up for future mutations.
        Will be called once on the first generation.
        Can be either implemented or left blank if this function is not desired.
        """
        pass

    def mutate_parameters(self):
        """
        Method to mutate the net's parameters.
        Will be called every iteration.
        Can be either implemented or left blank if this function is not desired.
        """
        pass

    def reset(self):
        """
        Method to reset the net's inner state.
        Can be either implemented or left blank if this function is not desired.
        """
        pass

    def setup_to_run(self):
        """
        Method to setup the net for it to then be ran.
        Can be either implemented or left blank if this function is not desired.
        """
        pass
        
    def setup_to_save(self):
        """
        Method to setup the net for it to be pickled (to a file or to be sent to another process).
        Can be either implemented or left blank if this function is not desired.
        """
        pass

    def __call__(self, x):
        """
        Run net for one timestep given input 'x'.
        Should be implemented.
        """
        raise NotImplementedError