import os
import logging

# Gale
from template.runner.base import BaseRunner

# Delegated
from .setup import GraphClassificationSetup
from .train import GraphClassificationTrain
from .evaluate import GraphClassificationEvaluate


class GraphClassification(BaseRunner):

    def __init__(self):
        """
        Attributes
        ----------
        setup = BaseSetup
            (strategy design pattern) Object responsible for setup operations
        """
        super().__init__()

        # ensure that only one GPU is used
        # TODO parallelize
        cuda_devices = os.environ['CUDA_VISIBLE_DEVICES']
        if len(cuda_devices) > 1:
            ind = cuda_devices[0]
            logging.warning("This runner can only be used on one GPU at a time, selecting GPU {}".format(ind))
            os.environ['CUDA_VISIBLE_DEVICES'] = ind

        self.setup = GraphClassificationSetup()

    def prepare(self, model_name, resume, batch_lrscheduler_name, epoch_lrscheduler_name, **kwargs) -> dict:
        """
        Override methods from BaseRunner

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
        train_loader : torch_geometric.data.DataLoader
        val_loader : torch_geometric.data.DataLoader
        test_loader : torch_geometric.data..DataLoader
            Train/Val/Test set dataloader
        optimizer : torch.optim
            Optimizer to use during training, e.g. SGD
        criterion : torch.nn.modules.loss
            Loss function to use, e.g. nll_loss
        batch_lr_schedulers : list(torch.optim.lr_scheduler)
            List of lr schedulers to be called after each batch. By default there is a warmup lr scheduler
        epoch_lr_schedulers : list(torch.optim.lr_scheduler)
            List of lr schedulers to be called after each epoch. Can be empty
        """
        # Setting up the dataloaders
        train_loader, val_loader, test_loader, num_classes, num_features = self.setup.set_up_dataloaders(**kwargs)

        # Setting up model, optimizer, criterion
        model = self.setup.setup_model(model_name=model_name, num_classes=num_classes, train_loader=train_loader,
                                       num_features=num_features, **kwargs)
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
            "num_features": num_features,
            "best_value": best_value,
            "train_loader": train_loader,
            "val_loader": val_loader,
            "test_loader": test_loader,
            "optimizer": optimizer,
            "criterion": criterion,
            "batch_lr_schedulers": batch_lr_schedulers,
            "epoch_lr_schedulers": epoch_lr_schedulers,
        }

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
        test_value = self._test(model=model, criterion=criterion, epoch=epochs - 1, current_log_folder=current_log_folder, **kwargs)
        logging.info(f'Test: {test_value}')
        logging.info('Training completed')
        return test_value

    ####################################################################################################################
    """
    These methods delegate their function to other classes in this package. 
    It is useful because sub-classes can selectively change the logic of certain parts only.
    """

    def _train(self, train_loader, **kwargs):
        return GraphClassificationTrain.run(data_loader=train_loader, logging_label='train', **kwargs)

    def _validate(self, val_loader, **kwargs):
        return GraphClassificationEvaluate.run(data_loader=val_loader, logging_label='val', **kwargs)

    def _test(self, test_loader, **kwargs):
        return GraphClassificationEvaluate.run(data_loader=test_loader, logging_label='test', **kwargs)
