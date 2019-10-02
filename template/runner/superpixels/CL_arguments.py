# Utils

# DeepDIVA
import argparse

from template.runner.base import BaseCLArguments

class CLArguments(BaseCLArguments):

    def __init__(self):
        # Create parser
        self.parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description='Template for training a network on a dataset')
        # Add all options
        self._general_parameters(self.parser)
        self._superpixel_options(self.parser)

    def _superpixel_options(self, parser):
        """ Superpixels options """

        parser_general = parser.add_argument_group('SUPERPIXELS', 'Superpixels Options')
        parser_general.add_argument("-i", "--img-filename",
                                    type=str,
                                    default=None)
        parser_general.add_argument("-n", "--nPixels-on-side",
                                    type=int,
                                    help="the desired number of pixels on the side of a superpixel",)
        parser_general.add_argument("--i-std",
                                    type=int,
                                    help="std dev for color Gaussians, should be 5<= value <=40. A smaller value leads to more irregular superpixels",)
        parser_general.add_argument('--sartorious_cell',
                                   default=False,
                                   action='store_true',
                                   help='Flag on whether the data should be preprocessed as a cell image or not')