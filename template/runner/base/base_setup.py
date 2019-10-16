# Utils
import inspect
import logging
import os
# Torch related stuff
import shutil
import sys
from abc import abstractmethod

import numpy as np
import pandas as pd
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data

import models
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

    ####################################################################################################################
    # General setup: model, optimizer, lr scheduler and criterion
    @classmethod
    def setup_model(cls, model_name, no_cuda, num_classes=None, load_model=None, strict=True, **kwargs):
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
        stric : bool
            Enforces key match between loaded state_dict and model definition

        Returns
        -------
        model : DataParallel
            The model
        """
        # Load saved model dictionary
        if load_model:
            if not os.path.isfile(load_model):
                logging.error("No model dict found at '{}'".format(load_model))
                raise SystemExit
            logging.info('Loading a saved model')
            checkpoint = torch.load(load_model, map_location=lambda storage, loc: storage)
            if 'model_name' in checkpoint:
                model_name = checkpoint['model_name']
            # Override the number of classes based on the size of the last layer in the dictionary
            if num_classes is None or type(num_classes) == int:
                num_classes = next(reversed(checkpoint['state_dict'].values())).shape[0]

        # Initialize the model
        logging.info('Setting up model {}'.format(model_name))
        model = models.__dict__[model_name](output_channels=num_classes, **kwargs)

        # Transfer model to GPU
        if not no_cuda:
            logging.info('Transfer model to GPU')
            model = torch.nn.DataParallel(model).cuda()
            cudnn.benchmark = True

        # Load saved model weights
        if load_model:
            try:
                model.load_state_dict(checkpoint['state_dict'], strict=strict)
            except Exception as exp:
                logging.warning(exp)

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

        params = {}
        # For all arguments declared in the constructor signature of the selected optimizer
        for p in inspect.getfullargspec(torch.optim.__dict__[optimizer_name].__init__).args:
            # Add it to a dictionary in case it exists a corresponding value in kwargs
            if p in kwargs:
                params.update({p: kwargs[p]})
        # Create an return the optimizer with the correct list of parameters
        return torch.optim.__dict__[optimizer_name](model.parameters(), **params)

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
        # Create an return the optimizer with the correct list of parameters
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
            except:
                logging.warning('Unable to load information for data balancing. Using normal criterion')

        if not no_cuda:
            criterion.cuda()
        return criterion

    ####################################################################################################################
    # Analytics handling
    @classmethod
    def _load_analytics_csv(cls, input_folder, **kwargs):
        """ Load the analytics.csv file. If it is missing, attempt creating it

        Parameters
        ----------
        input_folder : string
            Path string that points to the three folder train/val/test. Example: ~/../../data/svhn

        Returns
        -------
        file
            The csv file
        """
        # If analytics.csv file not present, run the analytics on the dataset
        if not os.path.exists(os.path.join(input_folder, "analytics.csv")):
            logging.warning('Missing analytics.csv file for dataset located at {}'.format(input_folder))
            try:
                logging.warning('Attempt creating analytics.csv file for dataset located at {}'.format(input_folder))
                cls.create_analytics_csv(input_folder=input_folder, **kwargs)
                logging.warning('Created analytics.csv file for dataset located at {} '.format(input_folder))
            except NotImplementedError:
                logging.error('The method create_analytics_csv() is not implemented.')
                sys.exit(-1)
            except:
                logging.error('Creation of analytics.csv failed.')
                raise SystemError
        # Loads the analytics csv
        return pd.read_csv(os.path.join(input_folder, "analytics.csv"), header=None)

    @classmethod
    @abstractmethod
    def create_analytics_csv(cls, input_folder, **kwargs):
        """
        Create the analytics.csv file at the location specified by dataset_folder

        Format of the file:
            file name: analytics.csv
            cat analytics.csv

            mean[RGB], float, float , float
            std[RGB], float, float, float
            class_weights[num_classes], float, float, float*[one for each class in the dataset]

        Parameters
        ----------
        input_folder : string
            Path string that points to the dataset location
        """
        raise NotImplementedError

    @classmethod
    def load_mean_std_from_file(cls, **kwargs):
        """ Recover mean and std from the analytics.csv file

        Returns
        -------
        ndarray[double], ndarray[double]
            Mean and Std of the selected dataset, contained in the analytics.csv file.
        """
        # Loads the analytics csv and extract mean and std
        csv_file = cls._load_analytics_csv(**kwargs)
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
    def load_class_weights_from_file(cls, **kwargs):
        """ Recover class weights from the analytics.csv file (weights are the inverse of frequency)

        Returns
        -------
        ndarray[double]
            Class weights for the selected dataset, contained in the analytics.csv file.
        """
        # Loads the analytics csv and extract mean and std
        csv_file = cls._load_analytics_csv(**kwargs)
        for row in csv_file.values:
            if 'weights' in str(row[0]).lower():
                weights = np.array([x for x in row[1:] if str(x) != 'nan'], dtype=float)
        if 'weights' not in locals():
            logging.error("Class weights not found in analytics.csv")
            raise EOFError
        return weights

    ####################################################################################################################
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
            pass
            # Split the data into train/val/test folders
            # TODO use darwin-py
            # cls.split_darwin_dataset(input_folder=input_folder, **kwargs)

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
                                                   num_workers=workers,
                                                   pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_ds,
                                                 batch_size=batch_size,
                                                 num_workers=workers,
                                                 pin_memory=True)
        test_loader = torch.utils.data.DataLoader(test_ds,
                                                  batch_size=batch_size,
                                                  num_workers=workers,
                                                  pin_memory=True)
        return train_loader, val_loader, test_loader

    ####################################################################################################################
    # Transforms handling
    @classmethod
    def set_up_transforms(cls, train_ds, val_ds, test_ds, **kwargs):
        """Set up the data transform for FoodIQ"""
        # Assign the transform to splits
        train_ds.transform = cls.get_train_transform(**kwargs)
        for ds in [val_ds, test_ds]:
            ds.transform = cls.get_test_transform(**kwargs)
        for ds in [train_ds, val_ds, test_ds]:
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

    ####################################################################################################################
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
        except:
            expected_input_size = None

        d = {
            'epoch': epoch + 1,
            'arch': str(type(model)),
            'expected_input_size': expected_input_size,
            'model_name': model_name,
            'state_dict': model.state_dict(),
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

    ####################################################################################################################
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
