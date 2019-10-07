import base64
from pathlib import Path

import requests
from gust import Gust

from template.RunMe import RunMe

gust = Gust()
gust.start()

@gust.load
def load_function(weights_url: str, params: dict):
    """Download the model and delegate to Gale the duty of loading it in memory

    Parameters
    ----------
    weights_url : str
        URL of the model weights to download and save on the file system
    params : Dict
        Dictionary containing all the parameters for Gale, in the Gale format e.g. "-rc" : "ImageClassification"

        NECESSARY ENTRIES
            "-rc" : runner class to use

    Returns
    -------
    Dict
        Gale response
    """
    # Download the model weights and provide a path to it
    params["--load-model"] = download_model(weights_url)
    return RunMe().start(["--pre-load", *_prepare_CLArguments(title='infer', params=params)])

@gust.inference
def infer_function(image:dict, params:dict):
    """Run inference with the model and image provided

    Parameters
    ----------
    image : Dict
        Dictionary containing the base64 image to do inference on
    params : Dict
        Dictionary containing all the parameters for Gale, in the Gale format e.g. "-rc" : "ImageClassification"

        NECESSARY ENTRIES
            "-rc" : runner class to use

    Returns
    -------
    Dict
        Gale response
    """
    # Image comes as a base64 string
    with open("/tmp/inference.png", "wb") as fh:
        fh.write(base64.decodebytes(image['base64']))
    params['--input-folder'] = "/tmp/inference.png"
    # Execute Gale
    return RunMe().start(_prepare_CLArguments(title='infer', params=params))

@gust.train
def train_function(dataset_id:str, params: dict):
    """Train a model with the parameters provided in data.

    Parameters
    ----------
    dataset_id: str
        ID of the dataset on Darwin
    params : Dict
        Dictionary containing all the parameters for Gale, in the Gale format e.g. "-rc" : "ImageClassification"

        NECESSARY ENTRIES
            "-rc" : runner class to use
    Returns
    -------
    Dict
        Gale response
    """
    # TODO Use Darwin-py to create a Pytorch Dataset
    print(f"I wish I could download {dataset_id} but my developers did not yet enabled me to.")
    return RunMe().start(_prepare_CLArguments(title='train', params=params))

###################################################################################################
def download_model(url: str):
    """Downloads a file given an url and returns its path on the current file system

    Parameters
    ----------
    url : str
        URL from which the file needs to be downloaded

    Returns
    -------
    path : str
        Path to the downloaded file
    """
    path = Path("/tmp/model.pth")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
    return path

def _prepare_CLArguments(title:str , params:dict):
    """Prepares all the CLArguments to be passed to Gale

    Parameters
    ----------
    title : str
        Either 'infer' or 'train'. Specifies which subroutine should be called.
    params : dict
        Dictionary containing all extra parameters received with the request

    Returns
    -------
    List(str)
        List of strings containing all the arguments to be passed to GALE
    """
    # As per Njord implementation 'data' can't be None, but better safe than sorry!
    assert params is not None
    # Inject default parameters to make the HTTP requests less verbose.
    default = _default_parameters(title)
    # Merge default with data arguments. Note: A default parameter WILL NOT OVERRIDE a parameter already present in 'data'
    params = {**default, **params}
    # Make sure all elements are strings (e.g could be int coming from data) and make a list out of the dict
    return [str(x) for k, v in params.items() for x in [k, v]]

def _default_parameters(title:str) -> dict:
    """Prepare default parameters depending from the train/inference type.

    Parameters
    ----------
    title : str
        Either 'infer' or 'train'. Specifies which subroutine should be called.

    Returns
    -------
    dict
        Returns a dictionary with the original data and the default parameters which were missing
    """
    if title == "infer":
        return {"--inference": "",
                "--input-folder": "NOT USED IN PRE-LOAD, OVERRIDE ME AT ACTUAL INFERENCE TIME",
                "--ignoregit": "",
                "--output-folder":  '/gale/output',
                "--experiment-name": "inference_gale",}

    if title == "train":
        return {"--disable-dataset-integrity": "",
                "--ignoregit": "",
                "--output-folder": '/gale/output',
                "--experiment-name": "train_gale",}

    raise ValueError(f"Invalid value of title ({title}).")