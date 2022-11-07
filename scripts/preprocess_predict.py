#!/usr/bin/env python3

import argparse
import glob
import sys
from collections import defaultdict
from pathlib import Path

sys.path.append("../")

from qdf.hyperparameters import BASIS_SET, GRID_INTERVAL, RADIUS
from qdf.preprocess import create_dataset, load_dict
from qdf.settings import DATASET_PATH

if __name__ == "__main__":

    # e.g python preprocess_predict.py --dataset_train=QM9under7atoms_homolumo_eV
    #                                  --dataset_predict=QM9full_homolumo_eV
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_trained", type=str, required=True)
    parser.add_argument("--dataset_predict", type=str, required=True)
    args = parser.parse_args()

    dataset_trained = args.dataset_trained
    dataset_predict = args.dataset_predict

    dir_trained = Path(DATASET_PATH, dataset_trained)
    dir_predict = Path(DATASET_PATH, dataset_predict)

    filename = Path(str(dir_trained), "orbitaldict_" + str(BASIS_SET) + ".pickle")
    orbital_dict = load_dict(filename)
    N_orbitals = len(orbital_dict)

    print(
        "Preprocess",
        str(dataset_predict),
        "dataset.\n" "The preprocessed dataset is saved in",
        str(dir_predict),
        "directory.\n"
        "If the dataset size is large, "
        "it takes a long time and consume storage.\n"
        "Wait for a while...",
    )
    print("-" * 50)

    create_dataset(
        str(dir_predict), "test", BASIS_SET, RADIUS, GRID_INTERVAL, orbital_dict
    )
    if N_orbitals < len(orbital_dict):
        print(
            "##################### Warning!!!!!! #####################\n"
            "The prediction dataset contains unknown atoms\n"
            "that did not appear in the training dataset.\n"
            "The parameters for these atoms have not been learned yet\n"
            "and must be randomly initialized at this time.\n"
            "Therefore, the prediction will be unreliable\n"
            "and we stop this process.\n"
            "#########################################################"
        )
    else:
        print("-" * 50)
        print("The preprocess has finished.")
