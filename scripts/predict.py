#!/usr/bin/env python3

import argparse
import pickle
from pathlib import Path
import sys, os
sys.path.append('../')

import torch

from qdf.settings import DATASET_PATH, SAVE_PATH
from qdf.hyperparameters import BATCH_SIZE, OPERATION, ITERATION, LR, LR_DECAY, STEP_SIZE
from qdf.hyperparameters import BASIS_SET, RADIUS, GRID_INTERVAL
from qdf.hyperparameters import DIM, LAYER_FUNCTIONAL, HIDDEN_HK, LAYER_FUNCTIONAL, LAYER_HK
from qdf.datasets import QDFDataset
from qdf.models import QuantumDeepField
from qdf.wrappers import Tester


if __name__ == "__main__":

    # e.g python predict.py --dataset_trained=QM9under7atoms_homolumo_eV 
    #                       --dataset_predict=QM9full_homolumo_eV
    #                       --model_path="/home/raul/git/QuantumDeepField_molecule/pretrained/model"
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_trained', type=str, required=True)
    parser.add_argument('--dataset_predict', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()
    
    # raise some warnings
    if args.num_workers == 1:
        print(f"\t Note: Selected --num_workers=1 is default, but there are {os.cpu_count()} available.")
    
    if not Path(args.model_path).exists():
        raise Exception(f"Path to pretrained model is wrong. Couldnt find any file [{args.model_path}]")

    # by default use gpu if available in the system, otherwise use cpu (slower)
    if (not args.device) or (args.device not in ["cuda", "cpu"]):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\t Using device: {device}")
    print('-'*50)

    dir_dataset = Path(DATASET_PATH, args.dataset_trained)
    dir_predict = Path(DATASET_PATH, args.dataset_predict)

    field = '_'.join([BASIS_SET, str(RADIUS) + 'sphere', str(GRID_INTERVAL) + 'grid/'])
    dataset_test = QDFDataset(str(dir_predict) + '/test_' + field)
    dataloader_test = torch.utils.data.DataLoader(
                        dataset_test, BATCH_SIZE, shuffle=False, num_workers=args.num_workers,
                        collate_fn=lambda xs: list(zip(*xs)), pin_memory=True)

    with open(str(dir_dataset) + '/orbitaldict_' + BASIS_SET + '.pickle', 'rb') as f:
        orbital_dict = pickle.load(f)
    N_orbitals = len(orbital_dict)

    N_output = len(dataset_test[0][-2][0])

    # load the model from path
    model = QuantumDeepField(device, N_orbitals,
                            DIM, LAYER_FUNCTIONAL, OPERATION, N_output,
                            HIDDEN_HK, LAYER_HK).to(device)
    model.load_state_dict(torch.load(args.model_path,
                                     map_location=device))
    tester = Tester(model)

    print('Start predicting for', str(dir_predict), 'dataset.\n'
          'using the pretrained model with', str(dir_dataset), 'dataset.\n'
          'The prediction result is saved in the output directory.\n'
          'Wait for a while...')

    MAE, prediction = tester.test(dataloader_test, time=True)
    filename = Path(SAVE_PATH, "new_prediction.txt")
    ctr = 0
    while filename.exists():
        ctr += 1
        filename = Path(SAVE_PATH, f"new_prediction_{ctr}.txt")
    tester.save_prediction(prediction, str(filename))

    print('MAE:', MAE)
    print('The prediction has finished.')
