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
