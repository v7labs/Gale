# Utils
import inspect
import logging
import os
from collections import OrderedDict
# Torch related stuff
import shutil
from abc import abstractmethod
from pathlib import Path
from torch.hub import load_state_dict_from_url

import numpy as np
import pandas as pd
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data

import models
from models import model_zoo
from datasets.util.dataset_integrity import verify_dataset_integrity


# DeepDIVA


class BaseSetup:
    """ Basic implementation of the setup methods.

    Some of the methods could be static despite being annotated as class methods.
    This is a deliberate choice s.t. we handle the case that in the future there
    will be a state (attributes) in a setup, or especially, in one of its sub-classes.

    Note that any method with a cls.* call inside it can't be static or the OOP strategy
    design pattern will be broken!
    """

    ################################################################################################
    # General setup: model, optimizer, lr scheduler and criterion
    @classmethod
    def setup_model(cls, model_name, no_cuda, num_classes=None, load_model=None, **kwargs):
        """Setup the model, load and move to GPU if necessary

        Parameters
        ----------
        model_name : string
            Name of the model
        no_cuda : bool
            Specify whether to use the GPU or not
        num_classes : int
            How many different classes there are in our problem. Used for loading the model.
        load_model : string
            Path to a saved model

        Returns
        -------
        model : DataParallel
            The model
        """
        # Load saved model dictionary
        if load_model:
            state_dict = None
            if not os.path.isfile(load_model):
                # If it is not a valid path it will try to get the checkpoint
                # from the model zoo
                model_key = model_name + "_" + load_model
                if model_key not in model_zoo:
                    logging.error(f"No model dict found at '{load_model}'")
                    raise SystemExit
                load_model = model_zoo[model_key]
            logging.info(f"Loading a model from {load_model}")
            if load_model.startswith("http"):
                state_dict = load_state_dict_from_url(load_model, progress=False)
            else:
                checkpoint = torch.load(load_model, map_location=lambda storage, loc: storage)
                # Check consistency with model_name
                if 'model_name' in checkpoint:
                    if model_name is None:
                        model_name = checkpoint["model_name"]
                    elif model_name != checkpoint["model_name"]:
                        raise ValueError(f"name of the model in checkpoint does not match with --model-name. "
                                         f"{checkpoint['model_name']} != {model_name}")
                # Override the number of classes based on the list of classes in
                # the checkpoint
                if num_classes is None:
                    num_classes = len(checkpoint["classes"])
                elif not isinstance(checkpoint["classes"], dict) and num_classes != len(checkpoint["classes"]):
                    raise ValueError(f"number of classes in checkpoint does not match with --num-classes. "
                                     f"{len(checkpoint['classes'])} != {num_classes}")

                if "state_dict" in checkpoint:
                    state_dict = checkpoint["state_dict"]
                elif "model" in checkpoint:  # Clicker legacy. Eventually should be removed
                    state_dict = checkpoint["model"]
                else:
                    raise AttributeError(f"Could not find state dictionary in checkpoint {load_model}")

            assert state_dict is not None, f"Could not load a state dictionary from {load_model}"

        # Initialize the model
        logging.info('Setting up model {}'.format(model_name))
        # For all arguments declared in the constructor signature of the
        # selected model
        args = {}
        for p in inspect.getfullargspec(models.__dict__[model_name]).args:
            # Add it to a dictionary in case it exists a corresponding value in kwargs
            if p in kwargs:
                args.update({p: kwargs[p]})
        args["num_classes"] = num_classes
        model = models.__dict__[model_name](**args)

        # Check for module.* name space
        if load_model:
            # When its trained on GPU modules will have 'module' in their name
            is_module_named = np.any([k.startswith('module') for k in state_dict.keys() if k])
            if is_module_named:
                # Remove module. prefix from keys
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    if k.startswith('module.'):
                        k = k.replace('module.', '')
                    new_state_dict[k] = v
                state_dict = new_state_dict

            # Load the weights into the model
            try:
                model.load_state_dict(state_dict)
            except Exception as exp:
                logging.warning(exp)

        # Transfer model to GPU
        if not no_cuda:
            logging.info('Transfer model to GPU')
            model = torch.nn.DataParallel(model).cuda()
            cudnn.benchmark = True

        return model

    @classmethod
    def get_optimizer(cls, optimizer_name, model, **kwargs):
        """
        This function serves as interface between the command line and the optimizer.
        In fact each optimizer has a different set of parameters and in this way one can just change the optimizer
        in his experiments just by changing the parameters passed to the entry point.

        Parameters
        ----------
        optimizer_name:
            Name of the optimizers. See: torch.optim for a list of possible values
        model:
            The model with which the training will be done
        kwargs:
            List of all arguments to be used to init the optimizer
        Returns
        -------
        torch.optim
            The optimizer initialized with the provided parameters
        """
        # Verify the optimizer exists
        assert optimizer_name in torch.optim.__dict__

        params = [p for p in model.parameters() if p.requires_grad]

        args = {}
        # For all arguments declared in the constructor signature of the selected optimizer
        for p in inspect.getfullargspec(torch.optim.__dict__[optimizer_name].__init__).args:
            # Add it to a dictionary in case it exists a corresponding value in kwargs
            if p in kwargs:
                args.update({p: kwargs[p]})

        return torch.optim.__dict__[optimizer_name](params, **args)

    @classmethod
    def get_lrscheduler(cls, lrscheduler_name, **kwargs):
        """
        This function serves as interface between the command line and the lr scheduler.
        In fact each lr scheduler has a different set of parameters and in this way one can just change
        the lr scheduler in his experiments just by changing the parameters passed to the entry point.

        Parameters
        ----------
        lrscheduler_name:
            Name of the lr scheduler. See: torch.optim.lr_scheduler for a list of possible values
        kwargs:
            List of all arguments to be used to init the optimizer
        Returns
        -------
        torch.optim.lr_scheduler
            The optimizer initialized with the provided parameters
        """
        # Verify the optimizer exists
        assert lrscheduler_name in torch.optim.lr_scheduler.__dict__

        params = {}
        # For all arguments declared in the constructor signature of the selected optimizer
        for p in inspect.getfullargspec(torch.optim.lr_scheduler.__dict__[lrscheduler_name].__init__).args:
            # Add it to a dictionary in case it exists a corresponding value in kwargs
            if p in kwargs:
                params.update({p: kwargs[p]})
        # Create and return the optimizer with the correct list of parameters
        return torch.optim.lr_scheduler.__dict__[lrscheduler_name](**params)

    @classmethod
    def get_criterion(cls, criterion_name, no_cuda, disable_databalancing, **kwargs):
        """
        This function serves as an interface between the command line and the criterion.

        Parameters
        ----------
        criterion_name : string
            Name of the criterion
        no_cuda : bool
            Specify whether to use the GPU or not
        disable_databalancing : boolean
            If True the criterion will not be fed with the class frequencies. Use with care.

        Returns
        -------
        torch.nn
            The initialized criterion

        """
        # Verify that the criterion exists
        assert criterion_name in torch.nn.__dict__

        args = {}
        # For all arguments declared in the constructor signature of the selected optimizer
        for p in inspect.getfullargspec(torch.nn.__dict__[criterion_name].__init__).args:
            # Add it to a dictionary in case it exists a corresponding value in kwargs
            if p in kwargs:
                args.update({p: kwargs[p]})

        # Instantiate the criterion
        criterion = torch.nn.__dict__[criterion_name](**args)

        if not disable_databalancing:
            try:
                logging.info('Loading weights for data balancing')
                weights = cls.load_class_weights_from_file(**kwargs)
                criterion.weight = torch.from_numpy(weights).type(torch.FloatTensor)
            except Exception:
                logging.warning('Unable to load information for data balancing. Using normal criterion')

        if not no_cuda:
            criterion.cuda()
        return criterion

    ################################################################################################
    # Analytics handling
    @classmethod
    def create_analytics_csv(cls, input_folder, darwin_dataset, train_ds, **kwargs):
        """
        Creates the analytics.csv file at the location specified by input_folder

        Format of the file:
            file name: analytics.csv
            cat analytics.csv

            mean[RGB], float, float , float
            std[RGB], float, float, float
            class_weights[num_classes], float, float, float*[one for each class in the dataset]

        Parameters
        ----------
        input_folder : str
            Path string that points to the dataset location
        darwin_dataset : bool
            Flag for using the darwin dataset class instead
        train_ds : data.Dataset
            Train split dataset
        """
        # If it already exists your job is done
        if (Path(input_folder) / "analytics.csv").is_file():
            return

        logging.warning(f'Missing analytics.csv file for dataset located at {input_folder}')
        logging.warning(f'Attempt creating analytics.csv file')

        # Measure mean and std on train images
        logging.info(f'Measuring mean and std on train images')
        if darwin_dataset:
            mean, std = train_ds.measure_mean_std()
        else:
            mean, std = cls._measure_mean_std(
                input_folder=input_folder, train_ds=train_ds, **kwargs
            )

        # Measure weights for class balancing
        logging.info(f'Measuring class wrights')
        if darwin_dataset:
            class_weights = train_ds.measure_weights()
        else:
            class_weights = cls._measure_weights(
                input_folder=input_folder, train_ds=train_ds, **kwargs
            )

        # Save results as CSV file in the dataset folder
        logging.info(f'Saving to analytics.csv')
        df = pd.DataFrame([mean, std, class_weights])
        df.index = ['mean[RGB]', 'std[RGB]', 'class_weights[num_classes]']
        df.to_csv(Path(input_folder) / 'analytics.csv', header=False)
        logging.warning(f'Created analytics.csv file for dataset located at {input_folder}')

    @classmethod
    def _measure_mean_std(cls, train_ds, **kwargs):
        """Computes mean and std of train images, given the train loader

        Parameters
        ----------
        train_ds : data.Dataset
            Train split dataset

        Returns
        -------
        mean : ndarray[double]
            Mean value (for each channel) of all pixels of the images in the input folder
        std : ndarray[double]
            Standard deviation (for each channel) of all pixels of the images in the input folder
        """
        raise NotImplementedError

    @classmethod
    def _measure_weights(cls, train_ds, **kwargs):
        """Computes the class balancing weights (not the frequencies!!) given the train loader

        Parameters
        ----------
        train_ds : data.Dataset
            Train split dataset

        Returns
        -------
        class_weights : ndarray[double]
            Weight for each class in the train set (one for each class)
        """
        raise NotImplementedError

    @classmethod
    def load_mean_std_from_file(cls, input_folder, **kwargs):
        """ Recover mean and std from the analytics.csv file

        Parameters
        ----------
        input_folder : str
            Path string that points to the three folder train/val/test. Example: ~/../../data/svhn

        Returns
        -------
        ndarray[double], ndarray[double]
            Mean and Std of the selected dataset, contained in the analytics.csv file.
        """
        # Loads the analytics csv
        if not (Path(input_folder) / "analytics.csv").exists():
            raise SystemError(f"Analytics file not found in '{input_folder}'")
        csv_file = pd.read_csv(Path(input_folder) / "analytics.csv", header=None)
        # Extract mean and std
        for row in csv_file.values:
            if 'mean' in str(row[0]).lower():
                mean = np.array(row[1:4], dtype=float)
            if 'std' in str(row[0]).lower():
                std = np.array(row[1:4], dtype=float)
        if 'mean' not in locals() or 'std' not in locals():
            logging.error("Mean or std not found in analytics.csv")
            raise EOFError
        return mean, std

    @classmethod
    def load_class_weights_from_file(cls, input_folder, **kwargs):
        """ Recover class weights from the analytics.csv file (weights are the inverse of frequency)

        Parameters
        ----------
        input_folder : str
            Path string that points to the three folder train/val/test. Example: ~/../../data/svhn

        Returns
        -------
        ndarray[double]
            Class weights for the selected dataset, contained in the analytics.csv file.
        """
        # Loads the analytics csv
        if not (Path(input_folder) / "analytics.csv").exists():
            raise SystemError(f"Analytics file not found in '{input_folder}'")
        csv_file = pd.read_csv(Path(input_folder) / "analytics.csv", header=None)
        # Extracts the weights
        for row in csv_file.values:
            if 'weights' in str(row[0]).lower():
                weights = np.array([x for x in row[1:] if str(x) != 'nan'], dtype=float)
        if 'weights' not in locals():
            logging.error("Class weights not found in analytics.csv")
            raise EOFError
        return weights

    ################################################################################################
    # Dataloaders handling
    @classmethod
    def set_up_dataloaders(cls, **kwargs):
        """ Set up the dataloaders for the specified datasets.

        Returns
        -------
        train_loader : torch.utils.data.DataLoader
        val_loader : torch.utils.data.DataLoader
        test_loader : torch.utils.data.DataLoader
            Dataloaders for train, val and test.
        int
            Number of classes for the model.
        """
        logging.info('Loading {} from:{}'.format(
            os.path.basename(os.path.normpath(kwargs['input_folder'])),
            kwargs['input_folder'])
        )

        # Load the datasets
        train_ds, val_ds, test_ds = cls._get_datasets(**kwargs)

        # Create the analytics csv
        cls.create_analytics_csv(train_ds=train_ds, **kwargs)

        # Setup transforms
        logging.info('Setting up transforms')
        cls.set_up_transforms(train_ds=train_ds, val_ds=val_ds, test_ds=test_ds, **kwargs)

        # Get the dataloaders
        train_loader, val_loader, test_loader = cls._dataloaders_from_datasets(train_ds=train_ds,
                                                                               val_ds=val_ds,
                                                                               test_ds=test_ds,
                                                                               **kwargs)
        logging.info("Dataset loaded successfully")

        verify_dataset_integrity(**kwargs)

        return train_loader, val_loader, test_loader, len(train_ds.classes)

    @classmethod
    def _get_datasets(cls, input_folder, darwin_dataset, **kwargs):
        """
        Loads the dataset from file system and provides the dataset splits for train validation and test

        Parameters
        ----------
        input_folder : string
            Path string that points to the dataset location
        darwin_dataset : bool
            Flag for using the darwin dataset class instead

        Returns
        -------
        train_ds : data.Dataset
        val_ds : data.Dataset
        test_ds : data.Dataset
            Train, validation and test splits
        """
        if darwin_dataset:
            return cls.get_darwin_datasets(input_folder=input_folder, **kwargs)

        if not os.path.isdir(input_folder):
            raise RuntimeError("Dataset folder not found at " + input_folder)

        train_dir = os.path.join(input_folder, 'train')
        if not os.path.isdir(train_dir):
            raise RuntimeError("Train folder not found in the dataset_folder=" + input_folder)
        train_ds = cls.get_split(path=train_dir, **kwargs)

        val_dir = os.path.join(input_folder, 'val')
        if not os.path.isdir(val_dir):
            raise RuntimeError("Val folder not found in the dataset_folder=" + input_folder)
        val_ds = cls.get_split(path=val_dir, **kwargs)

        test_dir = os.path.join(input_folder, 'test')
        if not os.path.isdir(test_dir):
            raise RuntimeError("Test folder not found in the dataset_folder=" + input_folder)
        test_ds = cls.get_split(path=test_dir, **kwargs)

        return train_ds, val_ds, test_ds

    @classmethod
    def get_darwin_datasets(cls, input_folder: Path, split_folder: Path, split_type: str, **kwargs):
        """
        Used darwin-py integration to loads the dataset from file system and provide
        the dataset splits for train validation and test

        Parameters
        ----------
        input_folder : Path
            Path string that points to the dataset location
        split_folder : Path
            Path to the folder containing the split txt files
        split_type : str
            Type of the split txt file to choose. Either ['random', 'tags', 'polygon']

        Returns
        -------
        train_ds : data.Dataset
        val_ds : data.Dataset
        test_ds : data.Dataset
            Train, validation and test splits
        """
        assert input_folder is not None
        input_folder = Path(input_folder)
        assert input_folder.exists()
        # Point to the full path split folder
        assert split_folder is not None
        split_folder = input_folder / "lists" / split_folder
        assert split_folder.exists()

        # Select classification datasets
        from darwin.torch.dataset import Dataset
        train_ds = Dataset(
            root=input_folder, split=split_folder / (split_type + "_train.txt")
        )
        val_ds = Dataset(
            root=input_folder, split=split_folder / (split_type + "_val.txt")
        )
        test_ds = Dataset(
            root=input_folder, split=split_folder / (split_type + "_test.txt")
        )
        return train_ds, val_ds, test_ds

    @classmethod
    def get_split(cls, split_folder, **kwargs):
        """ Loads a split from file system and provides the dataset

        Parameters
        ----------------
        split_folder : string
            Path to the dataset on the file System

        Returns
        -------
        data.Dataset
        """
        raise NotImplementedError

    @classmethod
    def _dataloaders_from_datasets(cls, batch_size, train_ds, val_ds, test_ds, workers, **kwargs):
        """
        This function creates (and returns) dataloader from datasets objects

        Parameters
        ----------
        batch_size : int
            The size of the mini batch
        train_ds : data.Dataset
        val_ds : data.Dataset
        test_ds : data.Dataset
            Train, validation and test splits
        workers:
            Number of workers to use to load the data.

        Returns
        -------
        train_loader : torch.utils.data.DataLoader
        val_loader : torch.utils.data.DataLoader
        test_loader : torch.utils.data.DataLoader
            The dataloaders for each split passed
        """
        # Setup dataloaders
        logging.debug('Setting up dataloaders')
        train_loader = torch.utils.data.DataLoader(train_ds,
                                                   shuffle=True,
                                                   batch_size=batch_size,
                                                   num_workers=workers)
        val_loader = torch.utils.data.DataLoader(val_ds,
                                                 batch_size=batch_size,
                                                 num_workers=workers)
        test_loader = torch.utils.data.DataLoader(test_ds,
                                                  batch_size=batch_size,
                                                  num_workers=workers)
        return train_loader, val_loader, test_loader

    ################################################################################################
    # Transforms handling
    @classmethod
    def set_up_transforms(cls, train_ds, val_ds, test_ds, **kwargs):
        """Set up the data transform for FoodIQ"""
        # Assign the transform to splits
        train_ds.transform = cls.get_train_transform(**kwargs)
        for ds in [val_ds, test_ds]:
            if ds is not None:
                ds.transform = cls.get_test_transform(**kwargs)
        for ds in [train_ds, val_ds, test_ds]:
            if ds is not None:
                ds.target_transform = cls.get_target_transform(**kwargs)

    @classmethod
    @abstractmethod
    def get_train_transform(cls, **kwargs):
        """Set up the data transform for training split """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def get_test_transform(cls, **kwargs):
        """Set up the data transform for the test split or inference"""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def get_target_transform(cls, **kwargs):
        """Set up the target transform for all splits"""
        raise NotImplementedError

    ################################################################################################
    # Checkpointing handling
    @classmethod
    def checkpoint(cls, epoch, new_value, best_value, log_dir, the_lower_the_better=None, checkpoint_all_epochs=False, **kwargs):
        """Saves the current training checkpoint and the best valued checkpoint to file.

        Parameters
        ----------
        epoch : int
            Current epoch, for logging purpose only.
        new_value : float
            Current value achieved by the model at this epoch.
            To be compared with 'best_value'.
        best_value : float
            Best value ever obtained (so the last checkpointed model).
            To be compared with 'new_value'.
        log_dir : str
            Output folder where to put the model.
        the_lower_the_better : bool
            Changes the scale such that smaller values are better than bigger values
            (useful when metric evaluted is error rate)
        checkpoint_all_epochs : bool
            If enabled, save checkpoint after every epoch.
        kwargs : dict
            Any additional arguments.
        Returns
        -------
        best_value : float
            Best value ever obtained.

        """
        is_best = new_value > best_value if not the_lower_the_better else new_value < best_value
        best_value = new_value if is_best else best_value

        filename = os.path.join(log_dir, 'checkpoint.pth')
        torch.save(cls._serialize_dict(epoch=epoch, best_value=best_value, **kwargs), filename)
        if is_best:
            shutil.copyfile(filename, os.path.join(log_dir, 'best.pth'))

        # If enabled, save all checkpoints with epoch number.
        if checkpoint_all_epochs:
            shutil.move(filename, os.path.join(log_dir, f'checkpoint_{epoch}.pth.tar'))
        return best_value

    @classmethod
    def _serialize_dict(cls, model, model_name, epoch, best_value, train_loader, val_loader, test_loader, optimizer,
                        epoch_lr_schedulers, batch_lr_schedulers, **kwargs):
        """Creates a dictionary with all the arguments passes as parameter for further saving it on file
        model : nn.Module
            The actual model
        model_name : str
            Name of the model (the class!)
        epoch : int
            Epoch of training
        best_value : float
            Specifies the best value obtained by the model.
        train_loader : torch.utils.data.DataLoader
        test_loader : torch.utils.data.DataLoader
            The dataloaders for train and test set. Used to extract and save the transformations.
        optimizer : torch.optim
            The optimizer for the model
        batch_lr_schedulers : list(torch.optim.lr_scheduler)
            List of lr schedulers to be called after each batch.
        epoch_lr_schedulers : list(torch.optim.lr_scheduler)
            List of lr schedulers to be called after each epoch.

        Returns
        -------
        d : dict
            Dictionary with all the elements to save
        """
        train_transform = train_loader.dataset.transform
        if test_loader is not None:
            test_transform = test_loader.dataset.transform
        else:
            test_transform = val_loader.dataset.transform

        try:
            classes = train_loader.dataset.classes
        except AttributeError:
            classes = None
        try:
            expected_input_size = model.module.expected_input_size
        except Exception:
            expected_input_size = None

        if isinstance(model, torch.nn.DataParallel):
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()

        d = {
            'epoch': epoch + 1,
            'arch': str(type(model)),
            'expected_input_size': expected_input_size,
            'model_name': model_name,
            'state_dict': state_dict,
            'best_value': best_value,
            'classes': classes,
            'train_transform': train_transform,
            'test_transform': test_transform,
        }
        if optimizer:
            d['optimizer'] = optimizer.state_dict()
        for lrs in [*epoch_lr_schedulers, *batch_lr_schedulers]:
            d[str(type(lrs).__name__)] = lrs.state_dict()
        return d

    @classmethod
    def resume_checkpoint(cls, model, resume, optimizer, epoch_lr_schedulers, batch_lr_schedulers, **kwargs):
        """ Resume from checkpoint

        Parameters
        ----------
        model : nn.Module
            The actual model
        resume : string
            Path to a saved checkpoint
        optimizer : torch.optim
            The optimizer for the model
        batch_lr_schedulers : list(torch.optim.lr_scheduler)
            List of lr schedulers to be called after each batch.
        epoch_lr_schedulers : list(torch.optim.lr_scheduler)
            List of lr schedulers to be called after each epoch.

        Returns
        -------
        best_value : float
            Specifies the former best value obtained by the model.
            Relevant only if you are resuming training.
        """
        if os.path.isfile(resume):
            logging.info(f"Loading checkpoint '{resume}'")
            checkpoint = torch.load(resume)
            best_value = checkpoint['best_value']
            model.load_state_dict(checkpoint['state_dict'])
            if optimizer:
                optimizer.load_state_dict(checkpoint['optimizer'])
            for lrs in [*epoch_lr_schedulers, *batch_lr_schedulers]:
                lrs.load_state_dict(checkpoint[str(type(lrs).__name__)])
            logging.info(f"Loaded checkpoint '{resume}' (epoch {checkpoint['epoch']})")
        else:
            logging.error(f"No checkpoint found at '{resume}'")
            raise SystemExit
        return best_value

    ################################################################################################
    # Learning rate handling
    @classmethod
    def warmup_lr_scheduler(cls, optimizer, warmup_iters, warmup_factor):
        """Standard warmup LR scheduler

        Parameters
        ----------
        optimizer : torch.optim
            The optimizer
        warmup_iters : int
            How many batches should be processed before reaching full temperature
        warmup_factor : float
            Magnitude of the scaling to apply (original LR * warmup_factor)

        Returns
        -------
        torch.optim.lr_scheduler.LambdaLR
            The scheduler object
        """
        def f(x):
            if x >= warmup_iters:
                return 1
            alpha = float(x) / warmup_iters
            return warmup_factor * (1 - alpha) + alpha

        return torch.optim.lr_scheduler.LambdaLR(optimizer, f)
