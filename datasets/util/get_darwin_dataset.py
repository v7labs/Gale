import argparse
from pathlib import Path
from typing import Optional

from darwin.client import Client
from darwin.dataset.utils import split_dataset
from darwin.torch import ClassificationDataset, InstanceSegmentationDataset, \
    SemanticSegmentationDataset


def get_darwin_dataset(
        *,
        team_slug: str,
        dataset_slug: Optional[str] = None,
        dataset_id: Optional[str] = None,
        projects_dir: Optional[str] = None,
        token: Optional[str] = None,
        config_path: Optional[Path] = None,
        email: Optional[str] = None,
        password: Optional[str] = None,
        val_percentage: Optional[float] = 0.1,
        test_percentage: Optional[float] = 0.2,
        force_resplit: Optional[bool] = False,
        split_seed: Optional[int] = 42
):
    """
    Download a Darwin dataset on the file system.
    It is possible to select the way to authenticate and the configuration of
    the split of the dataset

    Parameters
    ----------
    team_slug : str
        Slug of the team to select
    dataset_slug : str
        This is the dataset name with everything lower-case, removed specials characters and
        spaces are replaced by dashes, e.g., `bird-species`. This string is unique within a team
    projects_dir : Path
        Path where the client should be initialized from (aka the root path)
    token : str
        Access token used to auth a specific request. It has a time spans of roughly 8min. to
    config_path : str
        Path to a configuration file to use to create the client
    email : str
        Email of the Darwin user to use for the login
    password : str
        Password of the Darwin user to use for the login
    val_percentage : float
        Percentage of images used in the validation set
    test_percentage : float
        Percentage of images used in the test set
    force_resplit : bool
        Discard previous split and create a new one
    split_seed : in
        Fix seed for random split creation

    Returns
    -------
    splits : dict
        Keys are the different splits (random, tags, ...) and values are the relative file names
    """
    # Authenticate client. The priority of the cases is arbitrarily chosen and should actually not matter
    if email is not None and password is not None:
        client = Client.login(email=email, password=password, projects_dir=projects_dir)
    elif token is not None:
        client = Client.from_token(token=token, projects_dir=projects_dir)
    elif config_path is not None:
        client = Client.from_config(config_path=config_path)
    else:
        client = Client.default(projects_dir=projects_dir)

    # Select the desired team
    if team_slug is not None:
        client.set_team(slug=team_slug)
    # Get the remote dataset
    dataset = client.get_remote_dataset(slug=dataset_slug, dataset_id=dataset_id)
    # Download the data on the file system
    dataset.pull()
    # Split the dataset with the param required
    return split_dataset(
        dataset=dataset,
        val_percentage=val_percentage,
        test_percentage=test_percentage,
        force_resplit=force_resplit,
        split_seed=split_seed
    )


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='This script can be used to download a dataset from Darwin')
    parser.add_argument('--projects-dir',
                        help='Path to where the dataset will be downloaded',
                        default=None,
                        type=Path)
    parser.add_argument('--dataset-slug',
                        help='Dataset slug (see Darwin documentation)',
                        default=None,
                        type=str)
    parser.add_argument('--dataset-id',
                        help='Dataset ID (see Darwin documentation)',
                        default=None,
                        type=str)
    parser.add_argument('--team-slug',
                        help='Team slug (see Darwin documentation)',
                        default=None,
                        type=str)
    parser.add_argument('--token',
                        help='Token to authenticate the client',
                        default=None,
                        type=str)
    parser.add_argument('--config-path',
                        help='Path to the configuration file to authenticate the client',
                        default=None,
                        type=Path)
    parser.add_argument('--email',
                        help='User email, for auth',
                        default=None,
                        type=str)
    parser.add_argument('--password',
                        help='User pwd, for auth',
                        default=None,
                        type=str)
    parser.add_argument('--val-percentage',
                        help='User pwd, for auth',
                        default=0.1,
                        type=float)
    parser.add_argument('--test-percentage',
                        help='User pwd, for auth',
                        default=0.2,
                        type=float)
    parser.add_argument('--force-resplit',
                        help='User pwd, for auth',
                        default=False,
                        type=bool)
    parser.add_argument('--split-seed',
                        help='User pwd, for auth',
                        default=42,
                        type=int)
    args = parser.parse_args()

    # If experiment name is not set, ask for one
    if (args.password is None) and (args.email is not None):
        from getpass import getpass
        args.password = getpass()

    # Run the actual code
    get_darwin_dataset(**args.__dict__)
