import json
import numpy as np
import os
from importlib import import_module

def initialize_environment(args, rank, size):

    if os.path.splitext(args.additional_arguments)[1] == ".json":

        with open(args.additional_arguments) as json_file:
            args.additional_arguments = json.load(json_file)

    else:
        
        args.additional_arguments = json.loads(args.additional_arguments)

    return getattr( import_module( args.env_path.replace('/', '.').replace('.py', '') ), 'Env' )(args, rank, size)

def deterministic_set(x):

    set = list( dict.fromkeys(x) )

    return set

def find_sublist_index(element, list):

    for sublist_index, sublist in enumerate(list):
        if element in sublist:
            return sublist_index

def list_dict(*args):

    dict = {}

    for arg in args:
        dict[arg] = []

    return dict
