# Utils
import glob
import logging
import os
import sys

import numpy as np
import pandas as pd
import torchvision
import yaml
from torchvision.transforms import transforms
import json
from sklearn.model_selection import train_test_split

# Gale
from datasets.generic_image_folder_dataset import ImageFolderDataset
from datasets.util.dataset_analytics import compute_mean_std_graphs, get_class_weights_graphs
from template.runner.base.base_setup import BaseSetup
from datasets.iamdb_gxl_dataset import GxlDataset


class GraphClassificationSetup(BaseSetup):
    """
    Implementation of the setup methods for Graph Neural Networks and the IAMDB datasets.
    """

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

        # TODO implement this
        # verify_dataset_integrity(**kwargs)

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
            # Split the data into train/val/test folders
            cls.split_darwin_dataset(input_folder=input_folder, **kwargs)

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