# Utils
import glob
import logging
import os
import shutil

import numpy as np
import pandas as pd
import torchvision
import yaml
from torchvision.transforms import transforms
import json
from sklearn.model_selection import train_test_split

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
    def split_darwin_dataset(cls, input_folder, darwin_splits, current_log_folder, **kwargs):
        """Splits a GUST-Elixir datasets downloaded into train/val/test with symlinks

        STRATIFIED SPLITTING (taking into account classes)

        Parameters
        ----------
        input_folder : str
            Path to the dataset on the file System
        darwin_splits : list(float)
            Specifies the % of the train/val/test splits
        current_log_folder : string
            Path to where logs/checkpoints are saved
        """
        # Load annotations and images file names
        annotations_path = os.path.join(input_folder, 'annotations')
        assert os.path.exists(annotations_path)
        annotations_file_names = np.asarray(glob.glob(annotations_path + '/*.json'))
        images_path = os.path.join(input_folder, 'images')
        assert os.path.exists(images_path)
        images_file_names = np.asarray([f for ext in ['jpg', 'jpeg', 'png'] for f in glob.glob(images_path + '/*.' + ext)])
        assert len(annotations_file_names) == len(images_file_names)

        # Sort file names list. It is important otherwise there is no matching!
        annotations_file_names = np.sort(annotations_file_names)
        images_file_names = np.sort(images_file_names)

        # Extract labels from annotations
        def extract_tag_name(jfile):
            for annotation in jfile['annotations']:
                if 'tag' in annotation.keys():
                    return annotation['name'].replace(' ', '_').replace("'", "")
            return None

        labels = []
        empty_annotations = []
        for i, file_name in enumerate(annotations_file_names):
            with open(file_name) as json_file:
                tag_name = extract_tag_name(json.load(json_file))
                if tag_name is not None:
                    labels.append(tag_name)
                else:
                    logging.error(f"Annotation file ({file_name}) has no tags. Skipping.")
                    empty_annotations.append(i)

        # Remove empty files from the lists
        if len(empty_annotations) > 0:
            logging.warning(f"Skipping {len(empty_annotations)} files.")
            images_file_names = np.delete(images_file_names, empty_annotations)
            annotations_file_names = np.delete(annotations_file_names, empty_annotations)
        labels = np.asarray(labels)
        assert len(labels) == len(annotations_file_names)
        assert len(labels) == len(images_file_names)

        # Stratify
        assert np.sum(darwin_splits) == 100
        X_train, X_tmp, y_train, y_tmp = train_test_split(images_file_names, labels,
                                                          test_size=(darwin_splits[1]+darwin_splits[2])/100,
                                                          random_state=42,
                                                          stratify=labels)
        X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp,
                                                        test_size= darwin_splits[2] / (darwin_splits[1] + darwin_splits[2]),
                                                        random_state=42,
                                                        stratify=y_tmp)

        # Split annotations with symlinks
        def _split_with_symlinks(X, y, split_name, *, symbolic=False):
            for file_name, label in zip(X, y):
                # Create the folder
                path = os.path.join(input_folder, split_name, label)
                if not os.path.exists(path):
                    os.makedirs(path)
                # Verify if file exists and remove if necessary
                dst = os.path.join(path, os.path.basename(file_name))
                if os.path.exists(dst):
                    os.remove(dst)
                if symbolic:
                    # Make symlinks
                    os.symlink(file_name, dst)
                else:
                    # Copy the actual file
                    shutil.copy(file_name, dst)

        _split_with_symlinks(X_train, y_train, 'train')
        _split_with_symlinks(X_val, y_val, 'val')
        _split_with_symlinks(X_test, y_test, 'test')

        # Log the splits
        with open(os.path.join(current_log_folder, "splits.yaml"), "w") as stream:
            log = {
                'train': [os.path.basename(f) for f in X_train],
                'val': [os.path.basename(f) for f in X_val],
                'test': [os.path.basename(f) for f in X_test],
            }
            yaml.dump(log, stream)
        logging.info(f"Done splitting dataset")

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