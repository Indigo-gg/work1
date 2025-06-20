from .base_options import BaseOptions
from .multiclass_options import modify_commandline_options
import argparse
class TrainOptions:
    """This class includes training options"""

    def __init__(self):
        """Reset the class; indicates new options setup."""
        parser = argparse.ArgumentParser(description='Train a model')
        parser = BaseOptions().initialize(parser)
        parser = modify_commandline_options(parser)  # Add multi-class specific options
        
        # Training parameters
        parser.add_argument('--optim', type=str, default='adam', help='optimizer to use [sgd | adam]')
        parser.add_argument('--new_optim', action='store_true', help='Use new optimizer settings')
        parser.add_argument('--earlystop_epoch', type=int, default=15, help='Early stopping patience')
        parser.add_argument('--niter', type=int, default=100, help='Number of training iterations')
        
        self.parser = parser

    def parse(self, print_options=True):
        """Parse our options, create checkpoints directory suffix, and print/ save the options"""
        opt = self.parser.parse_args()
        opt.is_train = True
        opt.isTrain = True  # Add compatibility attribute
        return opt
