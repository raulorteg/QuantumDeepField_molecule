#!/usr/bin/env python3

import timeit

import numpy as np
import torch
import torch.optim as optim


class Trainer(object):
    """
    Trainer object wrapper. This object wraps the training loop for the model. It uses Adam optimizer and a exponential
    scheduler for the learning rate decay. That is LR_{t} = LR_{t-step_size}**{lr_decay}.
    :param torch.nn.Module model: Torch loaded model to be trained.
    :param float lr: Learning rate hyperparameter value.
    :param float lr_decay: Multiplicative factor of learning rate decay hyperparameter.
    :param int step_size: Period of activation of learning rate decay hyperparameter.
    """

    def __init__(
        self, model: torch.nn.Module, lr: float, lr_decay: float, step_size: int
    ):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr)
        self.scheduler = optim.lr_scheduler.StepLR(
            optimizer=self.optimizer, step_size=step_size, gamma=lr_decay
        )

    def optimize(self, loss: torch.tensor, optimizer: torch.optim.Adam) -> None:
        """
        Wrapping function for pytorch's routine of zeroing the gradients, computing the loss
        backpropagation and updating the optimizer with the new gradients.
        :param torch.tensor loss: Torch.tensors with the copmuted gradients on the loss.
        :param torch.optim.Adam optimizer: Wrapped optimzer to zero the gradients and update with ne backpropagation.
        """
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def train(self, dataloader: torch.utils.data.DataLoader) -> tuple:
        """
        Main method of the Train wrapper object. GIven the datalaoder object it performs a single pass on the whole dataloader (an epoch). For
        every batch of the data it computes first the loss on the target property (target='E', supervised) optimizes it, then computes the loss
        on the potentials (target='V', unsupervised) and optimizes again. Minimizes two loss functions in terms of E and V.

        :param torch.utils.data.DataLoader dataloader: Dataloader object containing all the training samples.
        :returns: The total losses on the supervised target and total losses on the unsupervised objective V for the whole epoch.
        :rtype: tuple
        """
        losses_E, losses_V = 0, 0
        for data in dataloader:

            # compute the loss on the supervised objective (E)
            loss_E = self.model.forward(data, train=True, target="E")
            self.optimize(loss_E, self.optimizer)
            losses_E += loss_E.item()

            # compute the loss on the unsupervised objective (V)
            loss_V = self.model.forward(data, train=True, target="V")
            self.optimize(loss_V, self.optimizer)
            losses_V += loss_V.item()

        self.scheduler.step()
        return losses_E, losses_V


class Tester(object):
    """
    Tester object wrapper. This object wraps the testing loop for the model.
    :param torch.nn.Module model: Torch loaded model to be trained.
    """

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.num_samples = None  # initialize the number of samples in test set counter

    def test(
        self, dataloader: torch.utils.data.DataLoader, time: bool = False
    ) -> tuple:
        """
        Main method of the Test wrapper object. Given the datalaoder object it performs a single pass on the whole dataloader (an epoch). For
        every batch of the data it computes the loss on the target property (target='E', supervised).

        :param torch.utils.data.DataLoader dataloader: Dataloader object containing all the testing samples.
        :param bool time: Boolean flag to be set ofor the option of timing the code execution (Defaults to false).
        :returns: Mean Absolute Error computed on the test set given and a summary string with the predicitons.
        :rtype: tuple
        """
        # if the number of total samples in the test dataset is not set the compute it
        # to avoid doing this at every epoch
        if not self.num_samples:
            self.num_samples = sum([len(data[0]) for data in dataloader])

        # initialize the buffers to save the results
        IDs, Es, Es_ = [], [], []

        # Inititalize the (running) Sum Absolute Error
        SAE = 0

        # initialize the timing clock only if time option True
        if time:
            start = timeit.default_timer()

        for i, data in enumerate(dataloader):

            # inference step on a batch, returns the ids of the molecules, true properties and predicted properties
            idx, E, E_ = self.model.forward(data)

            # compute the Total (sum) absolute error between true and target values
            # and sum it to the running total
            SAE += torch.sum(torch.abs(E - E_), 0)

            # append the results to the buffers
            IDs += list(idx)
            Es += E.tolist()
            Es_ += E_.tolist()

            # if timing set and its the first batch the print out estimated running time
            if (time is True) and (i == 0):
                time = timeit.default_timer() - start
                minutes = len(dataloader) * time / 60
                hours = int(minutes / 60)
                minutes = int(minutes - 60 * hours)
                print(
                    "The prediction will finish in about",
                    hours,
                    "hours",
                    minutes,
                    "minutes.",
                )

        # copmute the Mean absolute error MAE = SAE/N
        MAE = (SAE / self.num_samples).tolist()  # Mean absolute error.
        MAE = ",".join([str(m) for m in MAE])  # For homo and lumo.

        # Creates the summary of results to be printed into a file
        prediction = "ID\tCorrect\tPredict\tError\n"
        for idx, E, E_ in zip(IDs, Es, Es_):
            error = np.abs(np.array(E) - np.array(E_))
            error = ",".join([str(e) for e in error])
            E = ",".join([str(e) for e in E])
            E_ = ",".join([str(e) for e in E_])
            prediction += "\t".join([idx, E, E_, error]) + "\n"

        return MAE, prediction

    def save_result(self, result: str, filename: str) -> None:
        """
        Saves the results summary into a file specified by filename parameter. Note that if the file
        exists the new results will be appended.

        :param str result: string summary of results to be dumped into a file.
        :param str filename: string filename for the file where the results are to be saved.
        """
        with open(filename, "a") as f:
            f.write(result + "\n")

    def save_prediction(self, prediction: str, filename: str) -> None:
        """
        Saves the prediction summary into a file specified by filename parameter. Note that if the file
        exists the new results will OVERWRITE the previous predictions.

        :param str prediction: string summary of predictions to be dumped into a file.
        :param str filename: string filename for the file where the predictions are to be saved.
        """
        with open(filename, "w") as f:
            f.write(prediction)

    def save_model(self, model: torch.nn.Module, filename: str) -> None:
        """
        Saves the model in the filename given, Wraps torc.save method to save ONLY the
        state dictionary.
        :param torch.nn.Module model: Torch model for which the state dictionary is to be saved.
        :param str filename: name of the file where the torch state dictionary is to be saved.
        """
        torch.save(model.state_dict(), filename)
