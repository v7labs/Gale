"""
This file is the template for the boilerplate of train/test of a DNN for image classification

There are a lot of parameter which can be specified to modify the behaviour and they should be used
instead of hard-coding stuff.
"""
# Utils
import logging
import os
import sys
from abc import abstractmethod
import numpy as np
import models
from template.runner.base import AbstractRunner

# Delegated
from template.runner.base.base_setup import BaseSetup


class BaseRunner(AbstractRunner):

    def __init__(self):
        """
        Attributes
        ----------
        setup = BaseSetup
            (strategy design pattern) Object responsible for setup operations
        """
        self.setup = BaseSetup()

    def single_run(self, **kwargs):
        """
        This is the main routine where train(), validate() and test() are called.

        Returns
        -------
        train_value : ndarray[floats] of size (1, `epochs`)
            Accuracy values for train split
        val_value : ndarray[floats] of size (1, `epochs`+1)
            Accuracy values for validation split. NOTE that the last element is actually the first validation at epoch=-1
        test_value : float
            Accuracy value for test split
        """

        # Prepare
        dict = self.prepare(**kwargs)

        if not kwargs["test_only"]:
            # Train routine
            train_value, val_value = self.train_routine(**dict, **kwargs)

        # Test routine
        if dict["test_loader"] is not None:
            test_value = self.test_routine(**dict, **kwargs)
        else:
            test_value = None

        return {
            'train': train_value,
            'val': val_value,
            'test': test_value,
        }

    ####################################################################################################################

    def prepare(self, model_name, resume, batch_lrscheduler_name, epoch_lrscheduler_name, **kwargs) -> dict:
        """
        Loads and prepares the data, the optimizer and the criterion

        Parameters
        ----------
        model_name : str
            Name of the model. Used for loading the model.
        resume : str
            Path to a saved checkpoint
        batch_lrscheduler_name : list(str)
            List of names of lr schedulers to be called after each batch. Can be empty
        epoch_lrscheduler_name : list(str)
            List of names of lr schedulers to be called after each epoch. Can be empty

        Returns
        -------
        model : DataParallel
            The model to train
        num_classes : int
            The number of classes as returned by the set_up_dataloaders()
        best_value : float
            Best value of the model so far. Non-zero only in case of --resume being used
        train_loader : torch.utils.data.dataloader.DataLoader
        val_loader : torch.utils.data.dataloader.DataLoader
        test_loader : torch.utils.data.dataloader.DataLoader
            Train/Val/Test set dataloader
        optimizer : torch.optim
            Optimizer to use during training, e.g. SGD
        criterion : torch.nn.modules.loss
            Loss function to use, e.g. cross-entropy
        batch_lr_schedulers : list(torch.optim.lr_scheduler)
            List of lr schedulers to be called after each batch. By default there is a warmup lr scheduler
        epoch_lr_schedulers : list(torch.optim.lr_scheduler)
            List of lr schedulers to be called after each epoch. Can be empty
        """
        # Get the selected model input size
        model_expected_input_size = models.__dict__[model_name].expected_input_size
        if type(model_expected_input_size) is not tuple or len(model_expected_input_size) != 2:
            logging.error('Model {model_name} expected input size is not a tuple. '
                          'Received: {model_expected_input_size}'
                          .format(model_name=model_name,
                                  model_expected_input_size=model_expected_input_size))
            sys.exit(-1)
        logging.info('Model {} expects input size of {}'.format(model_name, model_expected_input_size))

        # Setting up the dataloaders
        train_loader, val_loader, test_loader, num_classes = self.setup.set_up_dataloaders(model_expected_input_size=model_expected_input_size,
                                                                                           **kwargs)

        # Setting up model, optimizer, criterion
        model = self.setup.setup_model(model_name=model_name, num_classes=num_classes, train_loader=train_loader, **kwargs)
        optimizer = self.setup.get_optimizer(model=model, **kwargs)
        criterion = self.setup.get_criterion(**kwargs)

        # Setup the lr schedulers for epochs and batch updates
        batch_lr_schedulers = [self.setup.get_lrscheduler(optimizer=optimizer, lrscheduler_name=name, **kwargs)
                               for name in batch_lrscheduler_name]
        # Append by default a warm-up learning rate scheduler setup on 1 epoch period
        batch_lr_schedulers.append(self.setup.warmup_lr_scheduler(optimizer=optimizer,
                                                                  warmup_factor=1. / 1000,
                                                                  warmup_iters=len(train_loader) - 1))
        epoch_lr_schedulers = [self.setup.get_lrscheduler(optimizer=optimizer, lrscheduler_name=name, **kwargs)
                               for name in epoch_lrscheduler_name]

        # Resume from checkpoint if necessary
        if resume:
            best_value = self.setup.resume_checkpoint(model=model,
                                                      optimizer=optimizer,
                                                      resume=resume,
                                                      batch_lr_schedulers=batch_lr_schedulers,
                                                      epoch_lr_schedulers=epoch_lr_schedulers,
                                                      **kwargs)
        else:
            best_value = 0.0

        return {
            "model": model,
            "num_classes": num_classes,
            "best_value": best_value,
            "train_loader": train_loader,
            "val_loader": val_loader,
            "test_loader": test_loader,
            "optimizer": optimizer,
            "criterion": criterion,
            "batch_lr_schedulers": batch_lr_schedulers,
            "epoch_lr_schedulers": epoch_lr_schedulers,
        }

    def train_routine(self, best_value, validation_interval, start_epoch, epochs, checkpoint_all_epochs,
                      current_log_folder, epoch_lr_schedulers,
                      **kwargs):
        """
        Performs the training and validation routines

        Parameters
        ----------
        best_value : float
            Best value of the model so far. Non-zero only in case of --resume being used
        validation_interval : int
            Run evaluation on validation set every N epochs
        start_epoch : int
            Int to initialize the starting epoch. Non-zero only in case of --resume being used
        epochs : int
            Number of epochs to train
        checkpoint_all_epochs : bool
            Save checkpoint at each epoch
        current_log_folder : string
            Path to where logs/checkpoints are saved
        epoch_lr_schedulers : list(torch.optim.lr_scheduler)
            List of lr schedulers to call step() on after every epoch

        kwargs : dict
            Any additional arguments.

        Returns
        -------
        train_value : ndarray[floats] of size (1, `epochs`)
            Accuracy values for train split
        val_value : ndarray[floats] of size (1, `epochs`+1)
            Accuracy values for validation split
        """
        logging.info('Starting...')
        val_value = np.zeros((epochs + 1 - start_epoch))
        train_value = np.zeros((epochs - start_epoch))

        # Validate before training
        val_value[-1] = self._validate(epoch=-1, **kwargs)
        logging.info(f'Val epoch [-]: {val_value[-1]:.2f}')
        for epoch in range(start_epoch, epochs):
            # Train
            train_value[epoch] = self._train(epoch=epoch, **kwargs)

            if epoch % validation_interval == 0:
                # Validate
                val_value[epoch] = self._validate(epoch=epoch, **kwargs)
                logging.info(f'Val epoch [{epoch}]: {val_value[epoch]:.2f}')
                # Checkpoint
                best_value = self.setup.checkpoint(epoch=epoch,
                                                   new_value=val_value[epoch],
                                                   best_value=best_value,
                                                   log_dir=current_log_folder,
                                                   epoch_lr_schedulers=epoch_lr_schedulers,
                                                   checkpoint_all_epochs=checkpoint_all_epochs,
                                                   **kwargs)

            # Update the LR according to the scheduler
            for lr_scheduler in epoch_lr_schedulers:
                lr_scheduler.step()

        logging.info('Training done')
        return train_value, val_value

    def test_routine(self, model,  criterion, epochs, current_log_folder,
                     **kwargs):
        """
        Load the best model according to the validation score (early stopping) and runs the test routine.

        Parameters
        ----------
        model : DataParallel
            The model to train
        criterion : torch.nn.modules.loss
            Loss function to use, e.g. cross-entropy
        epochs : int
            After how many epochs are we testing
        current_log_folder : string
            Path to where logs/checkpoints are saved

        Returns
        -------
        test_value : float
            Accuracy value for test split
        """

        # Load the best model before evaluating on the test set (early stopping)
        logging.info('Loading the best model before evaluating on the test set.')

        if kwargs["load_model"] is not None:
            if not os.path.exists(kwargs["load_model"]):
                logging.error(f"Could not find model {kwargs['load_model']}. Terminating.")
                raise SystemExit
        elif os.path.exists(os.path.join(current_log_folder, 'best.pth')):
            kwargs["load_model"] = os.path.join(current_log_folder, 'best.pth')
        else:
            logging.warning('File model_best.pth.tar not found in {}'.format(current_log_folder))
            logging.warning('Using checkpoint.pth.tar instead')
            if os.path.exists(os.path.join(current_log_folder, 'checkpoint.pth')):
                kwargs["load_model"] = os.path.join(current_log_folder, 'checkpoint.pth')
            else:
                logging.warning('File checkpoint.pth.tar not found in {}'.format(current_log_folder))
                logging.error('Both best.pth and checkpoint.pth are not not found in {}. Terminating.'
                              .format(current_log_folder))
                raise SystemExit

        model = self.setup.setup_model(**kwargs)

        # Test
        test_value = self._test(model=model, criterion=criterion, epoch=epochs - 1, **kwargs)
        logging.info(f'Test: {test_value}')
        logging.info('Training completed')
        return test_value

    ####################################################################################################################
    """
    These methods delegate their function to other classes in this package.
    It is useful because sub-classes can selectively change the logic of certain parts only.
    """

    @abstractmethod
    def _train(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def _validate(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def _test(self, **kwargs):
        raise NotImplementedError
