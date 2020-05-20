# Utils
import argparse

# Torch
import torch

# Gale
import models
from template.runner.base.base_CL_arguments import BaseCLArguments


class CLArguments(BaseCLArguments):
    def __init__(self):
        # Add additional options
        super(CLArguments, self).__init__()

        self._graph_neural_network_options()

    def _graph_neural_network_options(self):
        """ Options used to run graph neural networks"""
        parser_graph_neural_network = self.parser.add_argument_group('GNN', 'Graph Neural Network Options')
        parser_graph_neural_network.add_argument('--rebuild-dataset',
                                                 default=False,
                                                 action='store_true',
                                                 help='Set to False if you want to load the dataset from /processed')
        parser_graph_neural_network.add_argument('--ignore-coordinates',
                                                 default=False,
                                                 action='store_true',
                                                 help='Set if node positions should not be used as a feature (assumes the features are present'
                                                      'as "x" and "y" in the gxl file)')
        parser_graph_neural_network.add_argument('--center-coordinates',
                                                 default=False,
                                                 action='store_true',
                                                 help='Set if the coordinates should be centred (xy - xy(avg))')
        parser_graph_neural_network.add_argument('--features-to-use',
                                                 type=str,
                                                 help='Specify features that should be used like "NodeFeatureName1,NodeFeatureName2,EdgefeatureName1"')
        parser_graph_neural_network.add_argument('--disable-feature-norm',
                                                 default=False,
                                                 action='store_true',
                                                 help='Node and edge features are not normalized (default: z-normalization)')
        parser_graph_neural_network.add_argument('--nb-neurons',
                                                 type=int,
                                                 default=128,
                                                 help='Number of hidden representations per graph convolution layer')

