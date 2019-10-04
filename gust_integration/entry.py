from gust import Gust
from template.RunMe import RunMe
from pathlib import Path
import os

gust = Gust()
gust.start()

@gust.load
def load_function(data):
    """ TODO

    Parameters
    ----------
    data

    Returns
    -------

    """
    data
    a = f"--model-name {data['model']}"
    return RunMe().start(["--pre-load", _prepare_CLArguments(title='infer', data=data)])

@gust.inference
def infer_function(data):
    """ TODO

    Parameters
    ----------
    data

    Returns
    -------

    """
    # Override the input_folder to point to image_path if set (happens at inference time)
    """image_path : str
        Path where the image have been downloaded with filename too. Used only in inference; None when training"""
    if image_path is not None:
        data['--input-folder'] = image_path

    # Execute Gale
    return RunMe().start(_prepare_CLArguments(title='infer', data=data))

@gust.train
def train_function(data):
    """ TODO

    Parameters
    ----------
    data

    Returns
    -------

    """
    # TODO Use Jon's branch to create a Pytorch Dataset
    return RunMe().start(_prepare_CLArguments(title='train', data=data))

###################################################################################################
def download_file(url):
    """ TODO

    Parameters
    ----------
    url

    Returns
    -------

    """
    path = Path("/tmp") / urlparse(url).path
    path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    return path

def _prepare_CLArguments(title, data):
    """Prepares all the CLArguments to be passed to Gale

    Parameters
    ----------
    title : str
        Either 'infer' or 'train'. Specifies which subroutine should be called.
    data : dict
        Dictionary containing all extra parameters received with the request
    Returns
    -------
    config_args : List(str)
        List of strings containing all the arguments to be passed to GALE
    """
    # As per Njord implementation 'data' can't be None, but better safe than sorry!
    if data is None:
        data = {}
    # Inject default parameters to make the HTTP requests less verbose.
    data = _default_parameters(title)
    # If there is a config file, read its arguments and merge them with data arguments
    config_args = _read_config_args(title)
    config_args = _merge_default_and_data(config_args, data)
    # Make sure all elements are strings (e.g could be int coming from data)
    config_args = [str(i) for i in config_args]
    return config_args

def _default_parameters(title: str) -> dict:
    """Inject default parameters into data depending from the train/inference type.
    Note: A default parameter WILL NOT OVERRIDE a parameter already present in 'data'

    Parameters
    ----------
    title : str
        Either 'infer' or 'train'. Specifies which subroutine should be called.
    data : dict
        Dictionary containing all extra parameters received with the request
    Returns
    -------
    dict
        Returns a dictionary with the original data and the default parameters which were missing
    """
    if title == "infer":
        return {"--inference": "",
                "--ignoregit": "",
                "--load-model": os.path.join(self.path, self.experiment_id, 'best.pth'),
                "--input-folder": "NOT USED IN PRE-LOAD, OVERRIDE ME AT ACTUAL INFERENCE TIME",
                "--output-folder":  '/gale/output',
                "--experiment-name": f"inference_{self.experiment_id}",}

    if title == "train":
        return {"--ignoregit": "",
                "--disable-dataset-integrity": "",
                "--output-folder": '/gale/output',
                "--experiment-name": f"train_{self.experiment_id}",}

    raise ValueError(f"Invalid value of title ({title}).")

def _merge_default_and_data(default, data):
    """If data has been provided, add (and OVERRIDE!) the parameters from the default list

    Parameters
    ----------
    default : List(str)
        List of arguments parsed from the relevant part of the config file. Might be empty
    data : dict
        Dictionary containing all extra parameters received with the request and default ones
    Returns
    -------
    config_args : List(str)
        List of arguments after merging data and the original config_args.
    """
    for k, v in data.items():
        try:
            index = default.index(k)
            # The arg is already in the list. If applicable, we update its value
            if v != "":
                # If value is not empty the next element in the list must be a value and not a name
                assert not default[index + 1].startswith('-')
                # Update the value
                default[index + 1] = v
        except ValueError:
            # The arg is not in the list, so we append it
            default.append(k)
            if v != "":
                default.append(v)
    return default