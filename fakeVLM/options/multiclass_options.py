import argparse
import os
from util import get_model

# Add new command line arguments for multi-class classification
def modify_commandline_options(parser):
    """Add new arguments and modify existing ones."""
    parser.add_argument('--n_classes', type=int, default=2, help='Number of classes in the dataset')
    return parser