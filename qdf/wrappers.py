#!/usr/bin/env python3

import timeit

import numpy as np
import torch
import torch.optim as optim


class Trainer(object):
    def __init__(self, model, lr, lr_decay, step_size):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer,
                                                   step_size, lr_decay)

    def optimize(self, loss, optimizer):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def train(self, dataloader):
        """Minimize two loss functions in terms of E and V."""
        losses_E, losses_V = 0, 0
        for data in dataloader:
            loss_E = self.model.forward(data, train=True, target='E')
            self.optimize(loss_E, self.optimizer)
            losses_E += loss_E.item()
            loss_V = self.model.forward(data, train=True, target='V')
            self.optimize(loss_V, self.optimizer)
            losses_V += loss_V.item()
        self.scheduler.step()
        return losses_E, losses_V


class Tester(object):
    def __init__(self, model):
        self.model = model

    def test(self, dataloader, time=False):
        N = sum([len(data[0]) for data in dataloader])
        IDs, Es, Es_ = [], [], []
        SAE = 0  # Sum absolute error.
        start = timeit.default_timer()

        for i, data in enumerate(dataloader):
            idx, E, E_ = self.model.forward(data)
            SAE_batch = torch.sum(torch.abs(E - E_), 0)
            SAE += SAE_batch
            IDs += list(idx)
            Es += E.tolist()
            Es_ += E_.tolist()

            if (time is True and i == 0):
                time = timeit.default_timer() - start
                minutes = len(dataloader) * time / 60
                hours = int(minutes / 60)
                minutes = int(minutes - 60 * hours)
                print('The prediction will finish in about',
                      hours, 'hours', minutes, 'minutes.')

        MAE = (SAE/N).tolist()  # Mean absolute error.
        MAE = ','.join([str(m) for m in MAE])  # For homo and lumo.

        prediction = 'ID\tCorrect\tPredict\tError\n'
        for idx, E, E_ in zip(IDs, Es, Es_):
            error = np.abs(np.array(E) - np.array(E_))
            error = ','.join([str(e) for e in error])
            E = ','.join([str(e) for e in E])
            E_ = ','.join([str(e) for e in E_])
            prediction += '\t'.join([idx, E, E_, error]) + '\n'

        return MAE, prediction

    def save_result(self, result, filename):
        with open(filename, 'a') as f:
            f.write(result + '\n')

    def save_prediction(self, prediction, filename):
        with open(filename, 'w') as f:
            f.write(prediction)

    def save_model(self, model, filename):
        torch.save(model.state_dict(), filename)