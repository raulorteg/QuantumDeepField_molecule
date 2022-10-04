#!/usr/bin/env python3

import argparse
from collections import defaultdict
import pickle
from pathlib import Path
import sys
sys.path.append("..")

from qdf.settings import DATASET_PATH, SAVE_PATH
from qdf.hyperparameters import RADIUS, BASIS_SET, GRID_INTERVAL
from qdf.definitions import ATOMICNUMBER_DICT
from qdf.preprocess import *


if __name__ == "__main__":

    # argument parser, to parse the dataset to be used
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    args = parser.parse_args()

    # define the path to the dataset
    dir_dataset = Path(DATASET_PATH, args.dataset)

    """Initialize orbital_dict, in which
    each key is an orbital type and each value is its index.
    """
    orbital_dict = defaultdict(lambda: len(orbital_dict))

    print('Preprocess', args.dataset, 'dataset.\n'
          'The preprocessed dataset is saved in', dir_dataset, 'directory.\n'
          'If the dataset size is large, '
          'it takes a long time and consume storage.\n'
          'Wait for a while...')
    print('-'*50)

    print('Training dataset...')
    create_dataset(str(dir_dataset), 'train',
                   BASIS_SET, RADIUS, GRID_INTERVAL, orbital_dict)
    print('-'*50)

    print('Validation dataset...')
    create_dataset(str(dir_dataset), 'val',
                   BASIS_SET, RADIUS, GRID_INTERVAL, orbital_dict)
    print('-'*50)

    print('Test dataset...')
    create_dataset(str(dir_dataset), 'test',
                   BASIS_SET, RADIUS, GRID_INTERVAL, orbital_dict)
    print('-'*50)

    with open(str(dir_dataset) + '/orbitaldict_' + BASIS_SET + '.pickle', 'wb') as f:
        pickle.dump(dict(orbital_dict), f)

    print('The preprocess has finished.')
