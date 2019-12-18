# Utils
import argparse

# Torch
import torch

# DeepDIVA
import models


class BaseCLArguments:

    def __init__(self):
        # Create parser
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                              description='Template for training a network on a dataset')
        # Add all options
        self._general_parameters(self.parser)
        self._sigopt_options(self.parser)
        self._data_options(self.parser)
        self._training_options(self.parser)
        self._optimizer_options(self.parser)
        self._criterion_options(self.parser)
        self._system_options(self.parser)
        self._inference_parameters(self.parser)

    def parse_arguments(self, args=None):
        """ Parse the command line arguments provided

        Parameters
        ----------
        args : str
            None, if set its a string which encloses the CLI arguments
            e.g. "--runner-class image_classification --output-folder log --dataset-folder datasets/MNIST"

        Returns
        -------
        args : dict
            Dictionary with the parsed arguments
        parser : ArgumentParser
            Parser used to process the arguments
        """
        # Parse argument
        args = self.parser.parse_args(args)

        # If experiment name is not set, ask for one
        if args.experiment_name is None:
            args.experiment_name = input("Please enter an experiment name:")

        return args, self.parser

    def _general_parameters(self, parser):
        """ General options """
        parser_general = parser.add_argument_group('GENERAL', 'General Options')
        parser_general.add_argument('-rc', '--runner-class')  # Do not remove! See RunMe()._parse_arguments()
        parser_general.add_argument('--experiment-name',
                                    type=str,
                                    default=None,
                                    help='provide a meaningful and descriptive name to this run')
        parser_general.add_argument('--input-folder',
                                    help='location of the dataset on the machine e.g root/data',
                                    required=False)
        parser_general.add_argument('--input-image',
                                    help='an image to process, encoded in base64',
                                    required=False)
        parser_general.add_argument('--output-folder',
                                    default='./output/',
                                    help='where to save all output files.', )
        parser_general.add_argument('--quiet',
                                    default=False,
                                    action='store_true',
                                    help='Do not print to stdout (log only).')
        parser_general.add_argument('--debug',
                                    default=False,
                                    action='store_true',
                                    help='log debug level messages')
        parser_general.add_argument('--multi-run',
                                    type=int,
                                    default=None,
                                    help='run main N times with different random seeds')
        parser_general.add_argument('--ignoregit',
                                    action='store_true',
                                    help='Run irrespective of git status.')
        parser_general.add_argument('--seed',
                                    type=int,
                                    default=None,
                                    help='random seed')
        parser_general.add_argument('--test-only',
                                    default=False,
                                    action='store_true',
                                    help='Skips the training phase.')
        parser_general.add_argument('--visualize-results',
                                    action='store_true',
                                    help='Generates qualitative results from the validation set.')

    def _sigopt_options(self, parser):
        """ SigOpt options"""

        parser_sigopt = parser.add_argument_group('GENERAL', 'General Options')
        parser_sigopt.add_argument('--sig-opt',
                                   type=str,
                                   default=None,
                                   help='path to a JSON file containing sig_opt variables and sig_opt bounds.')
        parser_sigopt.add_argument('--sig-opt-token',
                                   type=str,
                                   default=None,
                                   help='place your SigOpt API token here.')
        parser_sigopt.add_argument('--sig-opt-runs',
                                   type=int,
                                   default=100,
                                   help='number of updates of SigOpt required')
        parser_sigopt.add_argument('--sig-opt-project',
                                   type=str,
                                   default=None,
                                   help='place your SigOpt project name ere.')

    def _system_options(self, parser):
        """ System options """
        parser_system = parser.add_argument_group('SYS', 'System Options')
        parser_system.add_argument('--gpu-id',
                                   type=int, nargs='*',
                                   default=None,
                                   help='which GPUs to use for training (use all by default)')
        parser_system.add_argument('--no-cuda',
                                   action='store_true',
                                   default=False,
                                   help='run on CPU')
        parser_system.add_argument('--device',
                                   default='cuda',
                                   help='selects device')
        parser_system.add_argument('--log-interval',
                                   type=int,
                                   default=10,
                                   help='print loss/accuracy every N batches')
        parser_system.add_argument('-j', '--workers',
                                   type=int,
                                   default=4,
                                   help='workers used for train/val loaders')

    def _data_options(self, parser):
        """ Defines all parameters relative to the data. """
        parser_data = parser.add_argument_group('DATA', 'Dataset Options')
        parser_data.add_argument('--inmem',
                                 default=False,
                                 action='store_true',
                                 help='attempt to load the entire image dataset in memory')
        parser_data.add_argument('--darwin-dataset',
                                 default=False,
                                 action='store_true',
                                 help='flag for using the darwin_dataset.py dataset')
        parser_data.add_argument("--darwin-splits",
                                 default=[70.0, 10.0, 20.0],
                                 type=float,
                                 nargs=3,
                                 help="specify the % of the train/val/test split. Use as --darwin-splits float float float", )
        parser_data.add_argument('--disable-databalancing',
                                 default=False,
                                 action='store_true',
                                 help='Suppress data balancing')
        parser_data.add_argument('--disable-dataset-integrity',
                                 default=False,
                                 action='store_true',
                                 help='Suppress the dataset integrity verification')
        parser_data.add_argument('--enable-deep-dataset-integrity',
                                 default=False,
                                 action='store_true',
                                 help='enable the deep dataset integrity verification')

    def _training_options(self, parser):
        """ Training options """
        # List of possible custom models already implemented
        # NOTE: If a model is missing and you get a argument parser error: check in the init file of models if its there!
        model_options = [name for name in models.__dict__ if callable(models.__dict__[name])]

        parser_train = parser.add_argument_group('TRAIN', 'Training Options')
        parser_train.add_argument('--model-name',
                                  type=str,
                                  choices=model_options,
                                  help='which model to use for training')
        parser_train.add_argument('-b', '--batch-size',
                                  dest='batch_size',
                                  type=int,
                                  default=64,
                                  help='input batch size for training')
        parser_train.add_argument('--epochs',
                                  type=int,
                                  default=5,
                                  help='how many epochs to train')
        parser_train.add_argument('--pretrained',
                                  action='store_true',
                                  default=False,
                                  help='use pretrained model. (Not applicable for all models)')
        parser_train.add_argument('--load-model',
                                  type=str,
                                  default=None,
                                  help='path to latest checkpoint')
        parser_train.add_argument('--resume',
                                  type=str,
                                  default=None,
                                  help='path to latest checkpoint')
        parser_train.add_argument('--start-epoch',
                                  type=int,
                                  metavar='N',
                                  default=0,
                                  help='manual epoch number (useful on restarts)')
        parser_train.add_argument('--validation-interval',
                                  type=int,
                                  default=1,
                                  help='run evaluation on validation set every N epochs')
        parser_train.add_argument('--checkpoint-all-epochs',
                                  action='store_true',
                                  default=False,
                                  help='make a checkpoint after every epoch')

    def _optimizer_options(self, parser):
        """ Options specific for optimizers """
        # List of possible optimizers already implemented in PyTorch
        optimizer_options = [name for name in torch.optim.__dict__ if callable(torch.optim.__dict__[name])]
        lrscheduler_options = [name for name in torch.optim.lr_scheduler.__dict__ if
                               callable(torch.optim.lr_scheduler.__dict__[name])]
        parser_optimizer = parser.add_argument_group('OPTIMIZER', 'Optimizer Options')

        parser_optimizer.add_argument('--optimizer-name',
                                      choices=optimizer_options,
                                      default='SGD',
                                      help='optimizer to be used for training')
        parser_optimizer.add_argument('--momentum',
                                      type=float,
                                      default=0.9,
                                      help='momentum (parameter for the optimizer)')
        parser_optimizer.add_argument('--dampening',
                                      type=float,
                                      default=0,
                                      help='dampening (parameter for the SGD)')
        parser_optimizer.add_argument('--wd', '--weight-decay',
                                      type=float,
                                      dest='weight_decay',
                                      default=0,
                                      help='weight_decay coefficient, also known as L2 regularization')
        parser_optimizer.add_argument('--lr',
                                      type=float,
                                      default=0.001,
                                      help='learning rate to be used for training')
        parser_optimizer.add_argument('--step-size',
                                      default=10,
                                      type=int,
                                      help='decrease lr every step-size epochs')
        parser_optimizer.add_argument('--milestones',
                                      type=int, nargs='+',
                                      default=[8, 11],
                                      help='decrease lr every at each given milestone epoch')
        parser_optimizer.add_argument('--gamma',
                                      default=0.1,
                                      type=float,
                                      help='decrease lr by a factor of lr-gamma')
        parser_optimizer.add_argument('--epoch-lrscheduler-name',
                                      choices=lrscheduler_options,
                                      default=[],
                                      nargs='+',
                                      help='learning rate schedulers to be called after every epoch')
        parser_optimizer.add_argument('--batch-lrscheduler-name',
                                      choices=lrscheduler_options,
                                      default=[],
                                      nargs='+',
                                      help='learning rate schedulers to be called after every batch')
        parser_optimizer.add_argument('--base-lr',
                                      type=float,
                                      default=0.0001,
                                      help='parameter for torch.optim.lr_scheduler.CyclicLR scheduler')
        parser_optimizer.add_argument('--max-lr',
                                      type=float,
                                      default=0.01,
                                      help='parameter for torch.optim.lr_scheduler.CyclicLR scheduler')

    def _criterion_options(self, parser):
        """ Options specific for optimizers """
        parser_optimizer = parser.add_argument_group('CRITERION', 'Criterion Options')
        parser_optimizer.add_argument('--criterion-name',
                                      default='CrossEntropyLoss',
                                      help='criterion to be used for training')

    def _inference_parameters(self, parser):
        """ General options """
        parser_inference = parser.add_argument_group('INFERENCE', 'Inference Options')
        parser_inference.add_argument('--pre-load',
                                      default=False,
                                      action='store_true',
                                      help='Use this to only load the model')
        parser_inference.add_argument('--inference',
                                      default=False,
                                      action='store_true',
                                      help='flag for calling the fast (and lightweight) methods of calling execute')

