# Utils
import os

import pandas as pd
import torchvision
from torchvision.transforms import transforms

# Gale
from datasets.generic_image_folder_dataset import ImageFolderDataset
from datasets.util.dataset_analytics import compute_mean_std, get_class_weights
from template.runner.base.base_setup import BaseSetup


class ImageClassificationSetup(BaseSetup):

    @classmethod
    def create_analytics_csv(cls, input_folder, **kwargs):
        """Creates the analytics.csv file

        Parameters
        ----------
        input_folder : string
            Path string that points to the three folder train/val/test. Example: ~/../../data/svhn

        """
        train_folder = os.path.join(input_folder, 'train')
        mean, std = compute_mean_std(input_folder =train_folder, **kwargs)
        class_weights = get_class_weights(input_folder =train_folder, **kwargs)

        # Save results as CSV file in the dataset folder
        df = pd.DataFrame([mean, std, class_weights])
        df.index = ['mean[RGB]', 'std[RGB]', 'class_weights[num_classes]']
        df.to_csv(os.path.join(input_folder, 'analytics.csv'), header=False)


    @classmethod
    def get_split(cls, **kwargs):
        """ Loads a split from file system and provides the dataset

        Returns
        -------
        torch.utils.data.Dataset
            Split at the chosen path
        """
        return ImageFolderDataset(**kwargs)

    @classmethod
    def get_train_transform(cls, model_expected_input_size, **kwargs):
        """Set up the data transform for image classification

        Parameters
        ----------
        model_expected_input_size : tuple
           Specify the height and width that the model expects.

        Returns
        -------
        transform : torchvision.transforms.transforms.Compose
           the data transform
        """

        # Loads the analytics csv and extract mean and std
        mean, std = cls.load_mean_std_from_file(**kwargs)
        transform = transforms.Compose([
            transforms.Resize(model_expected_input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        return transform

    @classmethod
    def get_test_transform(cls, **kwargs):
        """Set up the data transform for the test split or inference"""
        return cls.get_train_transform(**kwargs)

    @classmethod
    def get_target_transform(cls, **kwargs):
        """Set up the target transform for all splits"""
        return None