"""
This file is the main entry point of DeepV7.

We introduce DeepV7: a DL framework built on the ashes of DeepDIVA (https://diva-dia.github.io/DeepDIVAweb/)

authors: Michele Alberti
"""

# Utils
import argparse
import datetime
import importlib
import inspect
import json
import logging
import os
import random
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import traceback

import colorlog
import numpy as np
# Torch related stuff
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
# SigOpt
from sigopt import Connection

# DeepDIVA
from template.runner.base import AbstractRunner, BaseCLArguments
from util.TB_writer import TBWriter
from util.misc import get_all_files_in_folders_and_subfolders
from util.misc import to_capital_camel_case
########################################################################################################################
from visualization.mean_std_plot import plot_mean_std


class RunMe:
    """
    This class is used as entry point of DeepDIVA.
    There are three main scenarios for using the framework:

        - Single run: (classic) run an experiment once with the given parameters specified by
                      command line. This is typical usage scenario.

        - Multi run: this will run multiple times an experiment. It basically runs the `single run`
                     scenario multiple times and aggregates the results. This is particularly useful
                     to counter effects of randomness.

        - Optimize with SigOpt: this will start an hyper-parameter optimization search with the aid
                                of SigOpt (State-of-the-art Bayesian optimization tool). For more
                                info on how to use it see the tutorial page on:
                                https://diva-dia.github.io/DeepDIVAweb/articles.html
    """

    # Reference to the argument parser. Useful for accessing types of arguments later e.g. setup.set_up_logging()
    parser = None

    @classmethod
    def start(cls, args=None) -> dict:
        """
        Select the use case based on the command line arguments and delegate the execution
        to the most appropriate sub-routine

        Returns
        -------
        _ : dict
        A dictionary which contains the payload of the runner class
        """
        # Parse all command line arguments
        args, cls.parser = cls._parse_arguments(args)

        # Select the use case
        if getattr(args, 'sig_opt', None) is not None:
            return cls._run_sig_opt(**args.__dict__)
        else:
            if getattr(args, 'inference', None):
                return cls._inference_execute(**args.__dict__)
            else:
                return cls._execute(**args.__dict__)

    @classmethod
    def _run_sig_opt(cls, sig_opt, sig_opt_token, sig_opt_runs, sig_opt_project, **kwargs) -> dict:
        """
        This function creates a new SigOpt experiment and optimizes the selected parameters.

        SigOpt is a state-of-the-art Bayesian optimization tool. For more info on how to use
        it see the tutorial page on: https://diva-dia.github.io/DeepDIVAweb/articles.html

        Parameters
        ----------
        sig_opt : str
            Path to a JSON file containing sig_opt variables and sig_opt bounds.
        sig_opt_token : str
            SigOpt API token
        sig_opt_runs : int
            Number of updates of SigOpt required
        sig_opt_project : str
            SigOpt project name

        Returns
        -------
        {} : dict
            At the moment it is not necessary to return meaningful values from here
        """
        # Load parameters from file
        with open(sig_opt, 'r') as f:
            parameters = json.loads(f.read())

        # Put your SigOpt token here.
        if sig_opt_token is None:
            logging.error('Enter your SigOpt API token using --sig-opt-token')
            raise SystemExit
        else:
            conn = Connection(client_token=sig_opt_token)
            experiment = conn.experiments().create(
                name=kwargs['experiment_name'],
                parameters=parameters,
                observation_budget=sig_opt_runs,
                project=sig_opt_project,
            )

            logging.info("Created experiment: https://sigopt.com/experiment/" + experiment.id)
            for i in range(sig_opt_runs):
                # Get suggestion from SigOpt
                suggestion = conn.experiments(experiment.id).suggestions().create()
                params = suggestion.assignments

                # Override/inject CL arguments received from SigOpt
                for key in params:
                    if isinstance(kwargs[key], bool):
                        params[key] = params[key].lower() in ['true']
                    kwargs[key] = params[key]

                # Run
                _, _, score = cls._execute(**kwargs)

                # In case of multi-run the return type will be a list (otherwise is a single float)
                if type(score) == float:
                    score = [score]
                for item in score:
                    conn.experiments(experiment.id).observations().create(suggestion=suggestion.id, value=item)
        return {}

    ####################################################################################################################
    ####################################################################################################################
    ####################################################################################################################
    @classmethod
    def _execute(cls, ignoregit, runner_class, multi_run, quiet, **kwargs):
        """
        Run an experiment once with the given parameters specified by command line.
        This is typical usage scenario.

        Parameters
        ----------
        ignoregit : bool
            Flag for verify git status
        runner_class : str
            Specify which is the runner class to select
        multi_run : int
            If not None, indicates how many multiple runs needs to be done
        quiet : bool
            Specify whether to print log to console or only to text file

        Returns
        -------
        return_value : dict
            Dictionary with the return value of the runner
        """
        def get_all_concrete_subclasses(class_name):
            csc = set()  # concrete subclasses
            if not inspect.isabstract(class_name):
                csc.add(class_name)
            for c in class_name.__subclasses__():
                csc = csc.union(get_all_concrete_subclasses(c))
            return csc

        # Set up logging
        # Don't use args.output_folder as that breaks when using SigOpt
        unpacked_args = {'ignoregit': ignoregit, 'runner_class': runner_class, 'multi_run': multi_run, 'quiet': quiet}
        current_log_folder = cls._set_up_logging(parser=RunMe.parser, quiet=quiet, args_dict={**kwargs, **unpacked_args},
                                                 **kwargs)

        # Copy the code into the output folder
        cls._copy_code(output_folder=current_log_folder)

        # Check Git status to verify all local changes have been committed
        if ignoregit:
            logging.warning('Git status is ignored!')
        else:
            cls._verify_git_status(current_log_folder)

        # Set up execution environment. Specify CUDA_VISIBLE_DEVICES and seeds
        cls._set_up_env(**kwargs)
        # kwargs["device"] = torch.device(kwargs["device"] if torch.cuda.is_available() else "cpu")
        kwargs["device"] = torch.device('cuda' if torch.cuda.is_available() else "cpu")

        # Find all subclasses of AbstractRunner and BaseRunner and select the chosen one among them based on -rc
        sub_classes = get_all_concrete_subclasses(AbstractRunner)
        runner_class = [c for c in sub_classes if to_capital_camel_case(runner_class).lower() == c.__name__.lower()]
        assert len(runner_class) == 1
        runner_class = runner_class[0]
        # TODO: make this more elegant
        # repack the runner_class with the reference for the actual runner, not as a string
        unpacked_args['runner_class'] = runner_class

        logging.warning(f'Current working directory is: {os.getcwd()}')

        # Run the actual experiment
        start_time = time.time()
        try:
            if multi_run is not None:
                return_value = cls._multi_run(current_log_folder=current_log_folder, **unpacked_args, **kwargs)
            else:
                return_value = runner_class().single_run(current_log_folder=current_log_folder, **unpacked_args, **kwargs)
            logging.info(f'Time taken: {datetime.timedelta(seconds=time.time() - start_time)}')
        except Exception as exp:
            if quiet:
                print('Unhandled error: {}'.format(repr(exp)))
            logging.error('Unhandled error: %s' % repr(exp))
            logging.error(traceback.format_exc())
            logging.error('Execution finished with errors :(')
            raise SystemExit
        finally:
            # Free logging resources
            logging.shutdown()
            logging.getLogger().handlers = []
            TBWriter().close()
            print('All done! (Log files at {} )'.format(current_log_folder))
        return return_value

    @classmethod
    def _inference_execute(cls, runner_class, **kwargs):
        """Run a lightweight version of the execute with no logging and extra stuff

        ***NOTE***
            TBWriter() will NOT be initialized in this mode!!
            Logging will NOT be available
            The environment (CUDA devices and the link) is not setup (GUST takes care of it)

        Parameters
        ----------
            Specify which is the runner class to select

        Returns
        -------
        payload : dict
            Dictionary with the payload to return after inference
        """
        def get_all_concrete_subclasses(class_name):
            csc = set()  # concrete subclasses
            if not inspect.isabstract(class_name):
                csc.add(class_name)
            for c in class_name.__subclasses__():
                csc = csc.union(get_all_concrete_subclasses(c))
            return csc

        # Set up execution environment. Specify CUDA_VISIBLE_DEVICES and seeds
        cls._set_up_env(**kwargs)

        # Find all subclasses of AbstractRunner and BaseRunner and select the chosen one among them based on -rc
        sub_classes = get_all_concrete_subclasses(AbstractRunner)
        runner_class = [c for c in sub_classes if to_capital_camel_case(runner_class).lower() == c.__name__.lower()]
        assert len(runner_class) == 1
        runner_class = runner_class[0]

        # Run the actual experiment
        start_time = time.time()
        return_value = runner_class().single_run(**kwargs)
        print(f'Payload (RunMe.py): {return_value}')
        print(f'Time taken: {(time.time() - start_time) * 1000:.0f}ms')
        return return_value

    @classmethod
    def _multi_run(cls, runner_class, current_log_folder, multi_run, epochs, **kwargs):
        """
        Run multiple times an experiment and aggregates the results.
        This is particularly useful to counter effects of randomness.

        Here multiple runs with same parameters are executed and the results averaged.
        Additionally "variance shaded plots" gets to be generated and are visible not only
        on FS but also on tensorboard under 'IMAGES'.

        Parameters
        ----------
        runner_class : String
            This is necessary to know on which class should we run the experiments.  Default is runner.image_classification.image_classification
        current_log_folder : String
            Path to the output folder. Required for saving the raw data of the plots
            generated by the multi-run routine.
        multi_run : int
            If not None, indicates how many multiple runs needs to be done
        epochs : int
            Number of epochs for the training. Used for the shaded plots.

        Returns
        -------
        train_scores : ndarray[float] of size (n, `epochs`)
        val_scores : ndarray[float] of size (n, `epochs`+1)
        test_score : ndarray[float] of size (n)
            Train, Val and Test results for each run (n) and epoch
        """

        # Instantiate the scores tables which will stores the results.
        train_scores = np.zeros((multi_run, epochs))
        val_scores = np.zeros((multi_run, epochs + 1))
        test_scores = np.zeros(multi_run)

        # As many times as runs
        for i in range(multi_run):
            logging.info('Multi-Run: {} of {}'.format(i + 1, multi_run))
            performance = runner_class().single_run(run=i,
                                                    current_log_folder=current_log_folder,
                                                    multi_run=multi_run,
                                                    epochs=epochs,
                                                    **kwargs)
            train_scores[i, :], val_scores[i, :], test_scores[i] = (performance['train'], performance['val'], performance['test'])


            # Generate and add to tensorboard the shaded plot for train
            train_curve = plot_mean_std(arr=train_scores[:i + 1],
                                        suptitle='Multi-Run: Train',
                                        title='Runs: {}'.format(i + 1),
                                        xlabel='Epoch', ylabel='Score',
                                        ylim=100.0)
            TBWriter().save_image(tag='train_curve', image=train_curve, global_step=i)
            logging.info('Generated mean-variance plot for train')

            # Generate and add to tensorboard the shaded plot for va
            val_curve = plot_mean_std(x=(np.arange(epochs + 1) - 1),
                                      arr=np.roll(val_scores[:i + 1], axis=1, shift=1),
                                      suptitle='Multi-Run: Val',
                                      title='Runs: {}'.format(i + 1),
                                      xlabel='Epoch', ylabel='Score',
                                      ylim=100.0)
            TBWriter().save_image(tag='val_curve', image=val_curve, global_step=i)
            logging.info('Generated mean-variance plot for val')

        # Log results on disk
        np.save(os.path.join(current_log_folder, 'train_values.npy'), train_scores)
        np.save(os.path.join(current_log_folder, 'val_values.npy'), val_scores)
        logging.info('Multi-run values for test-mean:{} test-std: {}'.format(np.mean(test_scores), np.std(test_scores)))
        s = 'mean: {}\n\nstd: {}'.format(np.mean(test_scores), np.std(test_scores))
        TBWriter().add_text(tag='Performance average and std over {} runs\n'.format(multi_run),
                            text_string=s)

        return train_scores, val_scores, test_scores

    ####################################################################################################################
    @classmethod
    def _parse_arguments(cls, args):
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
        def get_runner_class_options() -> dict:
            """Get all the classes which extend from AbstractRunner, located in any of the runner packages"""
            rdir = os.path.join(os.path.dirname(__file__), 'runner')
            packages_in_runner = [name for name in os.listdir(rdir) if os.path.isdir(os.path.join(rdir, name))]
            packages_in_runner.remove('base')
            if '__pycache__' in packages_in_runner:
                packages_in_runner.remove('__pycache__')
            return {n: c
                    for pkg in packages_in_runner
                    for n, c in inspect.getmembers(importlib.import_module('template.runner.' + pkg), inspect.isclass)
                    if issubclass(c, AbstractRunner) and not inspect.isabstract(c)}

        # Parse the runner-class to be used
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description='Get Runner class')
        # List of possible custom runner class. A runner class is defined as a module in template.runner
        runner_class_options = get_runner_class_options()
        parser.add_argument('-rc', '--runner-class',
                            choices=runner_class_options.keys(),
                            required=True,
                            help='Select which runner class to use.')
        runner_class = parser.parse_known_args(args)[0].runner_class

        # Fetch the CLArguments for the specific runner class and parse the arguments or the base ones if none are specified
        package_name = inspect.getmodule(runner_class_options[runner_class]).__package__
        cla = [c for _, c in inspect.getmembers(importlib.import_module(package_name), inspect.isclass)
               if issubclass(c, BaseCLArguments)]
        if len(cla) == 0:
            # If no sub-classes are found use the base implementation
            cla = getattr(sys.modules["template.runner.base.base_CL_arguments"], 'BaseCLArguments')
        elif len(cla) == 1:
            # One sub-class found, all good, pick it
            cla = cla[0]
        else:
            # Multiple sub-classes found. Invalid situation. Abort
            print('Multiple sub-classes of BaseCLArguments found in package: {}.'
                  'There must be only one. Exiting'.format('template.runner.'+runner_class))
            raise SystemExit
        args, parser = cla().parse_arguments(args)

        # Inject the runner-class for compatibility
        vars(args)['runner_class'] = runner_class

        return args, parser

    @classmethod
    def _verify_git_status(cls, current_log_folder):
        local_changes = False
        try:
            output_directory = os.path.split(os.getcwd())[0]
            git_url = subprocess.check_output(["git", "config", "--get", "remote.origin.url"])
            git_branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"])
            git_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
            git_status = str(subprocess.check_output(["git", "status"]))

            logging.debug('DeepV7 directory is:'.format(output_directory))
            logging.info('Git origin URL is: {}'.format(str(git_url)))
            logging.info('Current branch and hash are: {}  {}'.format(str(git_branch), str(git_hash)))
            local_changes = "nothing to commit" not in git_status and \
                            "working directory clean" not in git_status
            if local_changes:
                logging.warning('Running with an unclean working tree branch!')
        except Exception as exp:
            logging.warning('Git error: {}'.format(exp))
            local_changes = True
        finally:
            if local_changes:
                logging.error('Errors when acquiring git status. Use --ignoregit to still run.\n'
                              'This happens when the git folder has not been found on the file system\n'
                              'or when the code is not the same as the last version on the repository.\n'
                              'If you are running on a remote machine make sure to sync the .git folder as well.')
                logging.error('Finished with errors. (Log files at {} )'.format(current_log_folder))
                logging.shutdown()
                raise SystemExit

    @classmethod
    def _set_up_logging(cls, parser, experiment_name, output_folder, quiet, args_dict, debug, **kwargs):
        """
        Set up a logger for the experiment

        Parameters
        ----------
        parser : parser
            The argument parser
        experiment_name : string
            Name of the experiment. If not specify, accepted from command line.
        output_folder : string
            Path to where all experiment logs are stored.
        quiet : bool
            Specify whether to print log to console or only to text file
        debug : bool
            Specify the logging level
        args_dict : dict
            Contains the entire argument dictionary specified via command line.

        Returns
        -------
        log_folder : String
            The final logging folder tree
        """
        LOG_FILE = 'logs.txt'

        # Recover dataset name
        dataset = os.path.basename(os.path.normpath(kwargs['input_folder'])) if kwargs['input_folder'] is not None else ""

        """
        We extract the TRAIN parameters names (such as model_name, lr, ... ) from the parser directly.
        This is a somewhat risky operation because we access _private_variables of parsers classes.
        However, within our context this can be regarded as safe.
        Shall we be wrong, a quick fix is writing a list of possible parameters such as:

            train_param_list = ['model_name','lr', ...]

        and manually maintain it (boring!).

        Resources:
        https://stackoverflow.com/questions/31519997/is-it-possible-to-only-parse-one-argument-groups-parameters-with-argparse
        """

        # Fetch all non-default parameters
        non_default_parameters = []
        for group in parser._action_groups[2:]:
            if group.title not in ['GENERAL', 'DATA']:
                for action in group._group_actions:
                    if (kwargs[action.dest] is not None) and (
                            kwargs[action.dest] != action.default) \
                            and action.dest != 'load_model' \
                            and action.dest != 'input_image':
                        non_default_parameters.append(str(action.dest) + "=" + str(kwargs[action.dest]))

        # Build up final logging folder tree with the non-default training parameters
        log_folder = os.path.join(*[output_folder, experiment_name, dataset, *non_default_parameters,
                                    '{}'.format(time.strftime('%d-%m-%y-%Hh-%Mm-%Ss'))])
        # Remove spaces and weird char for a path (there could be some in the args values)
        log_folder = log_folder.replace(' ', '').replace(',', '_')
        if not os.path.exists(log_folder):
            os.makedirs(log_folder)

        # Setup logging
        root = logging.getLogger()
        log_level = logging.DEBUG if debug else logging.INFO
        root.setLevel(log_level)
        log_format = "[%(asctime)s] [%(levelname)8s] --- %(message)s (%(filename)s:%(lineno)s)"
        date_format = '%Y-%m-%d %H:%M:%S'

        if os.isatty(2):
            cformat = '%(log_color)s' + log_format
            formatter = colorlog.ColoredFormatter(cformat, date_format,
                                                  log_colors={
                                                      'DEBUG': 'cyan',
                                                      'INFO': 'white',
                                                      'WARNING': 'yellow',
                                                      'ERROR': 'red',
                                                      'CRITICAL': 'red,bg_white',
                                                  })
        else:
            formatter = logging.Formatter(log_format, date_format)

        if not quiet:
            ch = logging.StreamHandler()
            ch.setFormatter(formatter)
            root.addHandler(ch)

        fh = logging.FileHandler(os.path.join(log_folder, LOG_FILE))
        fh.setFormatter(logging.Formatter(log_format, date_format))
        root.addHandler(fh)

        logging.info('Setup logging. Log file: {}'.format(os.path.join(log_folder, LOG_FILE)))

        # Save args to logs_folder
        logging.info('Arguments saved to: {}'.format(os.path.join(log_folder, 'args.txt')))
        with open(os.path.join(log_folder, 'args.txt'), 'w') as f:
            f.write(json.dumps(args_dict))

        # Save all environment packages to logs_folder
        environment_yml = os.path.join(log_folder, 'environment.yml')
        subprocess.call('conda env export > {}'.format(environment_yml), shell=True)

        # Define Tensorboard SummaryWriter
        logging.info('Initialize Tensorboard SummaryWriter')
        TBWriter().init(log_dir=log_folder)

        # Add all parameters to Tensorboard
        TBWriter().add_text(tag='Args', text_string=json.dumps(args_dict))

        return log_folder

    @classmethod
    def _copy_code(cls, output_folder):
        """
        Makes a tar file with DeepDIVA that exists during runtime.

        Parameters
        ----------
        output_folder : str
            Path to output directory

        Returns
        -------
            None
        """
        # All file extensions to be saved by copy-code.
        FILE_TYPES = ['.sh', '.py']

        # Get DeepDIVA root
        cwd = os.getcwd()
        dd_root = os.path.join(cwd.split('DeepDIVA')[0], 'DeepDIVA')

        files = get_all_files_in_folders_and_subfolders(dd_root)

        # Get all files types in DeepDIVA as specified in FILE_TYPES
        code_files = [item for item in files if item.endswith(tuple(FILE_TYPES))]

        tmp_dir = tempfile.mkdtemp()

        for item in code_files:
            dest = os.path.join(tmp_dir, 'DeepDIVA', item.split('DeepDIVA')[1][1:])
            if not os.path.exists(os.path.dirname(dest)):
                os.makedirs(os.path.dirname(dest))
            shutil.copy(item, dest)

        # TODO: make it save a zipfile instead of a tarfile.
        with tarfile.open(os.path.join(output_folder, 'DeepV7.tar.gz'), 'w:gz') as tar:
            tar.add(tmp_dir, arcname='DeepDIVA')

        # Clean up all temporary files
        shutil.rmtree(tmp_dir)

    @classmethod
    def _set_up_env(cls, gpu_id=None, seed=None, multi_run=None, no_cuda=None, **kwargs):
        """
        Set up the execution environment.

        Parameters
        ----------
        gpu_id : string
            Specify the GPUs to be used
        seed :    int
            Seed all possible seeds for deterministic run
        multi_run : int
            Number of runs over the same code to produce mean-variance graph.
        no_cuda : bool
            Specify whether to use the GPU or not

        Returns
        -------
            None
        """

        # Set visible GPUs
        # TODO this does not work since it has to be called before importing
        # torch
        if gpu_id is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(id) for id in gpu_id])

        # Check if GPU's are available
        gpu_available = torch.cuda.is_available()
        if not gpu_available and not no_cuda:
            logging.warning('There are no GPUs available on this system, or your NVIDIA drivers are outdated.')
            logging.warning('Switch to CPU only computation using --no-cuda.')
            raise SystemExit

        # Seed the random
        if seed is None:
            # If seed is not specified by user, select a random value for the seed and then log it.
            seed = np.random.randint(2 ** 32 - 1, )
            logging.info('Randomly chosen seed is: {}'.format(seed))
        else:
            try:
                assert multi_run is None
            except Exception:
                logging.error('Arguments for seed AND multi-run should not be active at the same time!')
                raise SystemExit

            # Disable CuDNN only if seed is specified by user. Otherwise we can assume that the user does not want to
            # sacrifice speed for deterministic behaviour.
            # TODO: Check if setting torch.backends.cudnn.deterministic=True will ensure deterministic behavior.
            # Initial tests show torch.backends.cudnn.deterministic=True does not work correctly.
            if not no_cuda:
                torch.backends.cudnn.enabled = False

        # Python
        random.seed(seed)

        # Numpy random
        np.random.seed(seed)

        # Torch random
        torch.manual_seed(seed)
        if not no_cuda:
            torch.cuda.manual_seed_all(seed)


########################################################################################################################
if __name__ == "__main__":
    RunMe().start()
