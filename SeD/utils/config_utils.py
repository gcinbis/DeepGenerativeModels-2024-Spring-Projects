import sys
import os
from importlib import import_module
from easydict import EasyDict

"""
    This function modified from the Genforce library: https://github.com/genforce/genforce
"""
def parse_config(config_file):
    """Parses configuration from python file."""
    assert os.path.isfile(config_file) # Check if the file exists
    directory = os.path.dirname(config_file) # Get the directory of the file 
    filename = os.path.basename(config_file) # Get the name of the file
    module_name, extension = os.path.splitext(filename) # Get the module name and extension
    assert extension == '.py' # Check if the extension is .py
    sys.path.insert(0, directory) # Insert the directory to the first index of the path
    module = import_module(module_name) # Import the module
    sys.path.pop(0) # Remove the directory from the path
    config = EasyDict() # Create an EasyDict object
    for key, value in module.__dict__.items():
        if key.startswith('__'):
            continue
        config[key] = value
    del sys.modules[module_name]
    return config