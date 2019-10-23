import glob
import importlib
import os
from .registry import MODEL_REGISTRY

# List all the modules in the models subdirectory
modules = []
# Find the path of 'models' while being robust to the cwd
path = '/'.join(os.path.relpath(__file__, os.getcwd()).split('/')[:-1])
# Scan all files recursively in subfolders
for file in glob.glob(path+'/**/*.py', recursive=True):
    # Skip non-models files
    if "__init__" not in file and "registry" not in file:
        # Make the path and filename match the string needed for importlib
        s = file.replace("/", ".").replace(".py", "")
        # Append only the part from models onward
        modules.append(s[s.index('models'):])

# Importing all the models which are annotated with @Model
for module in modules:
    importlib.import_module(module)

# for module in modules:
#     importlib.import_module(module)

# Expose all the models
for m in MODEL_REGISTRY:
    globals()[m] = MODEL_REGISTRY[m]


model_zoo = {
    "fcn_resnet50_coco":
        None,
    "fcn_resnet101_coco":
        "https://download.pytorch.org/models/fcn_resnet101_coco-7ecb50ca.pth",
    "deeplabv3_resnet50_coco":  # TODO: upload to AWS
        "/home/jon/.torch/deeplabv3_resnet50_coco.pth",
    "deeplabv3_resnet101_coco":
        "https://download.pytorch.org/models/deeplabv3_resnet101_coco-586e9e4e.pth",
    "deeplabv3_resnet50_openimages":
        None,
    "deeplabv3_resnet101_openimages":
        None,
    "deeplabv3plus_resnet50_coco":  # TODO: upload to AWS
        "/home/jon/.torch/deeplabv3plus_resnet50_coco.pth",
    'fasterrcnn_resnet50_fpn_coco':
        'https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth',
    'maskrcnn_resnet50_fpn_coco':
        'https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth'
}
