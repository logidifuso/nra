import torch.nn as nn

class StaticNetBase(nn.Module):
    """
    Static Net Base class.
    Static Net objects are PyTorch-based Deep Neural Networks.
    Subclasses need to be named *Net*.
    """
    def __init__(self):
        
        super().__init__()
        
        self.device = None
        
    def reset(self):
        """
        Method to reset the net's inner state.
        Can be either implemented or left blank if this function is not desired.
        """
        pass

    def setup_to_run(self):
        """
        Method to setup the net for it to then be ran.
        """
        self.pre_setup_to_run()

        self.to(self.device)

    def pre_setup_to_run(self):
        """
        Placeholder method that can be used for user-defined pre-run setup.
        Can be implemented to send tensors to the device among other things.
        """
        pass
        
    def setup_to_save(self):
        """
        Method to setup the net for it to be pickled (to a file or to be sent to another process).
        """
        self.pre_setup_to_save()
        
        self.to('cpu')

    def pre_setup_to_save(self):
        """
        Placeholder method that can be used for user-defined pre-run setup.
        Can be implemented to send tensors back to the CPU for saving among other things.
        """
        pass

    def forward(self, x):
        """
        Run net for one timestep given input 'x'.
        Should be implemented.
        """
        raise NotImplementedError