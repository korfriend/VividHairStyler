import os
import yaml
from argparse import Namespace

def open_yaml(path):
    base_name = '_base'
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    if base_name in data.keys() and os.path.exists(data[base_name]):
        base_data = open_yaml(data[base_name])
        base_data.update(data)
        return base_data
    
    return data

def parse_yaml(yml_path: 'str'):
    return Namespace(**open_yaml(yml_path))