# Utils
import argparse

# Torch
import torch

# Gale
import models
from template.runner.base.base_CL_arguments import BaseCLArguments


class GraphClassificationIamdbCLArguments(BaseCLArguments):
    def __init__(self):
        # Add additional options
        super(GraphClassificationIamdbCLArguments, self).__init__()

        self._graph_neural_network_options(self.parser)

    def _graph_neural_network_options(self, parser):
        """ Options used to run graph neural networks"""
        parser_graph_neural_network = parser.add_argument_group('GNN', 'Graph Neural Network Options')
        parser_graph_neural_network.add_argument('--rebuild-dataset',
                                                 default=False,
                                                 action='store_true',
                                                 help='Set to False if you want to load the dataset from /processed')
        # TODO
        # parser_graph_neural_network.add_argument('--categorical-features-json',
        #                                          default=None,
        #                                          help='If true categorical feature values are loaded from file "categorical_features.json"'
        #                                               ' in the input folder')
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
        # parser_graph_neural_network.add_argument('--no-empty-graphs',
        #                                          default=False,
        #                                          action='store_true',
        #                                          help='Does not allow empty graphs in the dataset (they are skipped)')
        parser_graph_neural_network.add_argument('--disable-feature-norm',
                                                 default=False,
                                                 action='store_true',
                                                 help='Node and edge features are not normalized (default: z-normalization)')
        parser_graph_neural_network.add_argument('--nb-neurons',
                                                 type=int,
                                                 default=128,
                                                 help='Number of hidden representations per graph convolution layer')

