# Utils
import os
from pathlib import Path

import torchvision
from torchvision.transforms import transforms

# Gale
from datasets.custom import OnlyImage
from datasets.generic_image_folder_dataset import ImageFolderDataset
from datasets.util.dataset_analytics import compute_mean_std, get_class_weights
from template.runner.base.base_setup import BaseSetup


class ImageClassificationSetup(BaseSetup):

    @classmethod
    def _measure_mean_std(cls, input_folder, **kwargs):
        """Computes mean and std of train images

        Parameters
        ----------
        input_folder : string
            Path string that points to the three folder train/val/test. Example: ~/../../data/svhn

        Returns
        -------
        mean : ndarray[double]
            Mean value (for each channel) of all pixels of the images in the input folder
        std : ndarray[double]
            Standard deviation (for each channel) of all pixels of the images in the input folder
        """
        return compute_mean_std(input_folder =os.path.join(input_folder, 'train'), **kwargs)

    @classmethod
    def _measure_weights(cls, input_folder, **kwargs):
        """Computes the class balancing weights (not the frequencies!!)

        Parameters
        ----------
        input_folder : string
            Path string that points to the three folder train/val/test. Example: ~/../../data/svhn

        Returns
        -------
        class_weights : ndarray[double]
            Weight for each class in the train set (one for each class)
        """
        return get_class_weights(input_folder =os.path.join(input_folder, 'train'), **kwargs)

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
            Type of the split txt file to choose.

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
        from darwin.torch.dataset import ClassificationDataset
        train_ds = ClassificationDataset(
            root=input_folder, split=split_folder / (split_type + "_train.txt")
        )
        val_ds = ClassificationDataset(
            root=input_folder, split=split_folder / (split_type + "_val.txt")
        )
        test_ds = ClassificationDataset(
            root=input_folder, split=split_folder / (split_type + "_test.txt")
        )
        return train_ds, val_ds, test_ds

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
        transform = OnlyImage(transforms.Compose([
            transforms.Resize(model_expected_input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ]))
        return transform

    @classmethod
    def get_test_transform(cls, **kwargs):
        """Set up the data transform for the test split or inference"""
        return cls.get_train_transform(**kwargs)

    @classmethod
    def get_target_transform(cls, **kwargs):
        """Set up the target transform for all splits"""
        return None