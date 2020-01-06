# Utils
import logging
import os
import inspect
import numpy as np
import pandas as pd

# torch
import torchvision
import torch_geometric
import torch.backends.cudnn as cudnn
import torch

# Gale
from datasets.util.dataset_analytics import compute_mean_std_graphs, get_class_weights_graphs
from template.runner.base.base_setup import BaseSetup
from datasets.iamdb_gxl_dataset import GxlDataset
import models


class GraphClassificationSetup(BaseSetup):
    """
    Implementation of the setup methods for Graph Neural Networks and the IAMDB datasets.
    """
    ####################################################################################################################
    # General setup: model, optimizer, lr scheduler and criterion
    @classmethod
    def setup_model(cls, model_name, no_cuda, num_classes=None, num_features=None, load_model=None, strict=True, **kwargs):
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
            checkpoint = torch.load(load_model, map_location=lambda storage, loc: storage.cuda())
            if 'model_name' in checkpoint:
                model_name = checkpoint['model_name']
            # Override the number of classes based on the size of the last layer in the dictionary
            if num_classes is None or type(num_classes) == int:
                num_classes = next(reversed(checkpoint['state_dict'].values())).shape[0]

        # Initialize the model
        logging.info('Setting up model {}'.format(model_name))
        model = models.__dict__[model_name](output_channels=num_classes, num_features=num_features, **kwargs)

        # Transfer model to GPU
        if not no_cuda:
            logging.info('Transfer model to GPU')
            # TODO: parallelize
            # model = torch.nn.DataParallel(model).cuda()
            model = model.to(torch.device('cuda:{}'.format(os.environ['CUDA_VISIBLE_DEVICES'])))
            cudnn.benchmark = True

        # Load saved model weights
        if load_model:
            try:
                model.load_state_dict(checkpoint['state_dict'], strict=strict)
            except Exception as exp:
                logging.warning(exp)

        return model

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
            # TODO parallelize
            criterion.cuda(device=torch.device('cuda:{}'.format(os.environ['CUDA_VISIBLE_DEVICES'])))
        return criterion

    ####################################################################################################################
    # Analytics handling

    @classmethod
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
        # Sanity check on the folder
        if not os.path.isdir(input_folder):
            logging.error(f"Folder {input_folder} does not exist")
            raise FileNotFoundError

        train_dataset = GxlDataset(input_folder, subset='train', **kwargs)
        mean_std_nodef, mean_std_edgef = compute_mean_std_graphs(dataset=train_dataset, **kwargs)
        class_weights = get_class_weights_graphs(dataset=train_dataset, **kwargs)

        nodef_names = train_dataset.config['node_feature_names']
        edgef_names = train_dataset.config['edge_feature_names']

        # Save results as CSV file in the dataset folder
        df = pd.DataFrame([nodef_names, mean_std_nodef['mean'], mean_std_nodef['std'],
                           edgef_names, mean_std_edgef['mean'], mean_std_edgef['std'],
                           class_weights])

        df.index = ['node features', 'mean[node feature]', 'std[node feature]',
                    'edge features', 'mean[edge feature]', 'std[edge feature]',
                    'class_weights[num_classes]']

        df.to_csv(os.path.join(input_folder, 'analytics.csv'), header=False)

    @classmethod
    def load_class_weights_from_file(cls, **kwargs):
        """ Recover class weights from the analytics.csv file

        Returns
        -------
        ndarray[double]
            Class weights of the selected dataset, contained in the analytics.csv file.
        """
        # Loads the analytics csv and extract mean and std
        csv_file = cls._load_analytics_csv(**kwargs)
        return csv_file[csv_file[0] == 'class_weights[num_classes]'].values.tolist()[0][1:]

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
        mean_std = {'node_features': {}, 'edge_features': {}}
        for row in csv_file.values:
            if 'node' in str(row[0]).lower():
                flag = 'node_features'
            elif 'edge' in str(row[0]).lower():
                flag = 'edge_features'

            locals()
            # if flag in locals():
            if 'mean' in str(row[0]).lower():
                mean_std[flag]['mean'] = np.array([x for x in row[1:] if str(x) != 'nan'], dtype=float)
            if 'std' in str(row[0]).lower():
                mean_std[flag]['std'] = np.array([x for x in row[1:] if str(x) != 'nan'], dtype=float)

        for k, v in mean_std.items():
            if len(v['mean']) != len(v['std']):
                print("Number of {} does not match for mean and std in analytics.csv".format(k))
                raise EOFError
            if len(v['mean']) == 0 and len(v['std']) == 0:
                print("No mean and std for {} in analytics.csv".format(k))

        return mean_std

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

        # Loads the analytics csv and extract mean and std
        mean_std = cls.load_mean_std_from_file(**kwargs)

        # Load the datasets
        train_ds, val_ds, test_ds = cls._get_datasets(mean_std=mean_std, **kwargs)

        # Setup transforms
        # TODO implement this?
#        logging.info('Setting up transforms')
#        cls.set_up_transforms(train_ds=train_ds, val_ds=val_ds, test_ds=test_ds, **kwargs)

        # Get the dataloaders
        train_loader, val_loader, test_loader = cls._dataloaders_from_datasets(train_ds=train_ds,
                                                                               val_ds=val_ds,
                                                                               test_ds=test_ds,
                                                                               **kwargs)
        logging.info("Dataset loaded successfully")

        # TODO implement this
        # verify_dataset_integrity(**kwargs)
        return train_loader, val_loader, test_loader, train_ds.num_classes, train_ds.num_features

    @classmethod
    def _get_datasets(cls, input_folder, rebuild_dataset, **kwargs):
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

        if not os.path.isdir(input_folder):
            raise RuntimeError("Dataset folder not found at " + input_folder)

        data_dir = os.path.join(input_folder)
        if not os.path.isdir(data_dir):
            raise RuntimeError("Data folder not found in the dataset_folder=" + input_folder)

        if rebuild_dataset:
            logging.warning('Datasets will be rebuilt!')
        else:
            logging.warning('Datasets will NOT be rebuilt!')

        train_ds = cls.get_split(root_path=data_dir, subset='train', rebuild_dataset=rebuild_dataset, **kwargs)
        val_ds = cls.get_split(root_path=data_dir, subset='valid', rebuild_dataset=rebuild_dataset, **kwargs)
        test_ds = cls.get_split(root_path=data_dir, subset='test', rebuild_dataset=rebuild_dataset, **kwargs)

        return train_ds, val_ds, test_ds

    @classmethod
    def get_split(cls, **kwargs):
        """ Loads a split from file system and provides the dataset

        Returns
        -------
        torch_geometric.data.InMemoryDataset
        """
        return GxlDataset(**kwargs)

    @classmethod
    def _dataloaders_from_datasets(cls, batch_size, train_ds, val_ds, test_ds, **kwargs):
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
        train_loader : torch_geometric.data.DataLoader
        val_loader : torch_geometric.data.DataLoader
        test_loader : torch_geometric.data.DataLoader
            The dataloaders for each split passed
        """
        # Setup dataloaders
        logging.debug('Setting up dataloaders')
        train_loader = torch_geometric.data.DataLoader(train_ds,
                                                       batch_size=batch_size)
        val_loader = torch_geometric.data.DataLoader(val_ds,
                                                     batch_size=batch_size)
        test_loader = torch_geometric.data.DataLoader(test_ds,
                                                      batch_size=batch_size)
        return train_loader, val_loader, test_loader

    ####################################################################################################################
    # Transforms handling

    @classmethod
    def get_train_transform(cls, **kwargs):
        """Set up the data transform for image classification

        Parameters
        ----------

        Returns
        -------
        transform : torchvision.transforms.transforms.Compose
           the data transform
        """
        return None

    @classmethod
    def get_test_transform(cls, **kwargs):
        """Set up the data transform for the test split or inference"""
        return cls.get_train_transform(**kwargs)

    @classmethod
    def get_target_transform(cls, **kwargs):
        """Set up the target transform for all splits"""
        return None
