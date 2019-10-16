import argparse
from pathlib import Path

from darwin.client import Client
from darwin.dataset.utils import split_dataset


def get_darwin_dataset(output_path:str, dataset_slug:str, email:str, password:str):

    # Initialize client
    client =  Client.login(email=email,
                           password=password,
                           projects_dir_str=output_path)

    dataset = client.get_remote_dataset(slug=dataset_slug)
    dataset.pull()

    split_dataset(dataset=dataset,
                  val_percentage=0.1,
                  test_percentage=0.2,
                  force_resplit=True,
                  split_seed=42)

    print("done!")


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='This script can be used to download a dataset from Darwin')
    parser.add_argument('--output-path',
                        help='Path to where the dataset will be downloaded',
                        required=True,
                        type=Path,)
    parser.add_argument('--dataset-slug',
                        help='Dataset slug (see Darwin documentation)',
                        required=True,
                        type=str,)
    parser.add_argument('--email',
                        help='User email, for auth',
                        required=True,
                        type=str,)
    parser.add_argument('--password',
                        help='User pwd, for auth',
                        type=str,)
    args = parser.parse_args()

    # If experiment name is not set, ask for one
    if args.password is None:
        from getpass import getpass
        args.password = getpass()

    # Run the actual code
    get_darwin_dataset(**args.__dict__)



