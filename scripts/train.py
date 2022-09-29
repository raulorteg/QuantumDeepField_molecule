#!/usr/bin/env python3

import argparse
import os
from pathlib import Path
import pickle
import timeit

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

from qdf.hyperparameters import RADIUS, BASIS_SET, GRID_INTERVAL
from qdf.hyperparameters import DIM, LAYER_FUNCTIONAL, HIDDEN_HK, LAYER_HK, OPERATION
from qdf.hyperparameters import BATCH_SIZE, LR, LR_DECAY, STEP_SIZE, ITERATION


if __name__ == "__main__":

    """Args."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('setting')
    parser.add_argument('--num_workers', type=int, default=1)
    args = parser.parse_args()

    # raise some warnings
    if args.num_workers == 1:
        print(f"\t Note: Selected --num_workers=1 is default, but there are {os.cpu_count()} available, consider increasing the number to speed up the training.")
    
    unit = '(' + args.dataset.split('_')[-1] + ')'

    setting = args.setting

    """Fix the random seed (with the taxicab number)."""
    torch.manual_seed(1729)

    """GPU or CPU."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\t Using device: {device}")
    print('-'*50)

    """Create the dataloaders of training, val, and test set."""
    dir_dataset = '../dataset/' + args.dataset + '/'
    field = '_'.join([BASIS_SET, RADIUS + 'sphere', GRID_INTERVAL + 'grid/'])
    dataset_train = MyDataset(dir_dataset + 'train_' + field)
    dataset_val = MyDataset(dir_dataset + 'val_' + field)
    dataset_test = MyDataset(dir_dataset + 'test_' + field)
    dataloader_train = mydataloader(dataset_train, BATCH_SIZE, args.num_workers,
                                    shuffle=True)
    dataloader_val = mydataloader(dataset_val, BATCH_SIZE, args.num_workers)
    dataloader_test = mydataloader(dataset_test, BATCH_SIZE, args.num_workers)
    print('# of training samples: ', len(dataset_train))
    print('# of validation samples: ', len(dataset_val))
    print('# of test samples: ', len(dataset_test))
    print('-'*50)

    """Load orbital_dict generated in preprocessing."""
    with open(dir_dataset + 'orbitaldict_' + BASIS_SET + '.pickle', 'rb') as f:
        orbital_dict = pickle.load(f)
    N_orbitals = len(orbital_dict)

    """The output dimension in regression.
    When we learn only the atomization energy, N_output=1;
    when we learn the HOMO and LUMO simultaneously, N_output=2.
    """
    N_output = len(dataset_test[0][-2][0])

    print('Set a QDF model.')
    model = QuantumDeepField(device, N_orbitals,
                             DIM, LAYER_FUNCTIONAL, OPERATION, N_output,
                             HIDDEN_HK, LAYER_HK).to(device)
    trainer = Trainer(model, LR, LR_DECAY, STEP_SIZE)
    tester = Tester(model)
    print('# of model parameters:',
          sum([np.prod(p.size()) for p in model.parameters()]))
    print('-'*50)

    """Output files."""
    file_result = '../output/result--' + setting + '.txt'
    result = ('Epoch\tTime(sec)\tLoss_E\tLoss_V\t'
              'MAE_val' + unit + '\tMAE_test' + unit)
    with open(file_result, 'w') as f:
        f.write(result + '\n')
    file_prediction = '../output/prediction--' + setting + '.txt'
    file_model = '../output/model--' + setting

    print('Start training of the QDF model with', args.dataset, 'dataset.\n'
          'The training result is displayed in this terminal every epoch.\n'
          'The result, prediction, and trained model '
          'are saved in the output directory.\n'
          'Wait for a while...')

    start = timeit.default_timer()

    for epoch in range(ITERATION):
        loss_E, loss_V = trainer.train(dataloader_train)
        MAE_val = tester.test(dataloader_val)[0]
        MAE_test, prediction = tester.test(dataloader_test)
        time = timeit.default_timer() - start

        if epoch == 0:
            minutes = ITERATION * time / 60
            hours = int(minutes / 60)
            minutes = int(minutes - 60 * hours)
            print('The training will finish in about',
                  hours, 'hours', minutes, 'minutes.')
            print('-'*50)
            print(result)

        result = '\t'.join(map(str, [epoch, time, loss_E, loss_V,
                                     MAE_val, MAE_test]))
        tester.save_result(result, file_result)
        tester.save_prediction(prediction, file_prediction)
        tester.save_model(model, file_model)
        print(result)

    print('The training has finished.')
