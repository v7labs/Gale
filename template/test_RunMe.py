"""
Warning: LONG RUNTIME TESTS!

This test suite is designed to verify that the main components of the framework are not broken.
It is expected that smaller components or sub-parts are tested individually.

As we all know this will probably never happen, we will at least verify that the overall features
are correct and fully functional. These tests will take long time to run and are not supposed to
be run frequently. Nevertheless, it is important that before a PR or a push on the master branch
the main functions can be tested.

Please keep the list of these tests up to date as soon as you add new features.
"""
import shutil

import numpy as np
import pytest
from pathlib import Path

from datasets.util.get_a_dataset import cifar10
from template.RunMe import RunMe

INPUT_PATH = Path().absolute() / 'test_data'
OUTPUT_PATH = Path().absolute() / 'test_output'

@pytest.fixture(autouse=True)
def run_around_tests():
    # Prepare the test dataset
    print("Preparing data")
    INPUT_PATH.mkdir(exist_ok=True)
    OUTPUT_PATH.mkdir(exist_ok=True)
    cifar10(output_folder=INPUT_PATH)
    # A test function will be run at this point
    print("Running test")
    yield
    # Remove the test dataset and outputs
    shutil.rmtree(INPUT_PATH)
    shutil.rmtree(OUTPUT_PATH)
    print("Done!")

def test_one():
    """
    - Verify the sizes of the return of execute
    - Image classification with default parameters
    """
    epochs = 2
    args = ["-rc", "ImageClassification",
            "--experiment-name", "test_image_classification",
            "--ignoregit",
            "--input-folder", str(INPUT_PATH / "CIFAR10"),
            "--output-folder", str(OUTPUT_PATH),
            "--model-name", "CNN_basic",
            "--seed", "42",
            "--epochs", str(epochs)]

    payload = RunMe().start(args=args)

    # Verify sizes of the return values from execute
    assert len(payload.keys()) == 3
    assert 'train' in payload
    assert 'val' in payload
    assert 'test' in payload

    assert payload['train'] is not None
    assert len(payload['train'].shape) == 1
    assert payload['train'].shape[0] == epochs

    assert payload['val'] is not None
    assert len(payload['val'].shape) == 1
    assert payload['val'].shape[0] == epochs + 1

    assert payload['test'] is not None
    assert isinstance(payload['test'], float)


