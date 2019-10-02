import inspect
import os
import time
from ._generic_exceptions import *
from ._class_files_dirs import FilesDirs
from ._class_bunch import Bunch

 
def print_me(x):
    print(x)
    
def print_iterable(x):
    """
    Will print items in differnet lines.
    """
    map(print_me,x)     
 
_dirname_of_this_file = os.path.dirname(inspect.getfile(inspect.currentframe()))
