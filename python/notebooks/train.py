import os
import sys
import copy
import torch
import matplotlib.pyplot as plt
import numpy as np

import optuna

from tqdm import tqdm
from pathlib import Path
print(os.getcwd())
sys.path.append(str(Path('../src').resolve()))

from implicits import continuous_diff

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_model = None
        self.best_epoch = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model, epoch):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose: self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_epoch = epoch
            self.save_checkpoint(val_loss, model)
            self.best_model = copy.deepcopy(model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def train_siren(model, dset_train, dset_val, device, epochs,
                learning_rate, batch_size, loss_name,
                val_interval=10, trial=None, xy_cont=None, lam_tv=None, clip_grad=True):

    print(learning_rate, batch_size, loss_name)
    # training on a subset of the full dataset
    dload_train = torch.utils.data.DataLoader(dset_train, shuffle=True, batch_size=batch_size)
    dload_val = torch.utils.data.DataLoader(dset_val, shuffle=False, batch_size=len(dset_val))

    losses = {
        'MSE' : torch.nn.MSELoss(),
        'MAE' : torch.nn.L1Loss(),    
    }

    early_stopping = EarlyStopping(patience=5, delta=0.01)

    optim = torch.optim.Adam(lr=learning_rate, params=model.parameters())
    loss_fun = losses[loss_name]
            
    loss_train_track = []
    loss_val_track = []
    metrics_track = []

    model = model.to(device)
    model.train()


    train_bar = tqdm(range(epochs))
    best_val_loss = 100000
    for epoch in train_bar:

        # Training
        running_loss = 0
        for batch_idx, (coords, reference) in enumerate(dload_train):

            # model with some prior knowledge
            estimated = model(coords)
            train_loss = loss_fun(estimated, reference)

            if not xy_cont is None:
                # off-grid regularization with continuous total variation norm
                ind = torch.randint(0, xy_cont.shape[0], size=(coords.shape[0] // 2,1))
                dz_dxy = continuous_diff(xy_cont[ind,:], model)
                tv_norm = torch.mean(torch.abs(dz_dxy))
                train_loss = train_loss + lam_tv * tv_norm
                # print(train_loss, tv_norm, lam_tv * tv_norm)
    
            optim.zero_grad()
            train_loss.backward()
            if clip_grad:
                if isinstance(clip_grad, bool):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
            optim.step()

            
            running_loss = running_loss + train_loss.item()


        # Validation
        with torch.no_grad():
            coords, reference = next(iter(dload_val))
            estimated = model(coords)
            val_loss = loss_fun(estimated, reference)
            # metrics = compute_metrics(estimated, reference)
            mse = torch.mean(torch.abs(reference - estimated)**2).cpu().detach().numpy()
            metrics = {
                'mse' : mse
            }
            monitor_loss = mse

        metrics_track.append(metrics)
        loss_val_track.append(val_loss.item())
        loss_train_track.append(train_loss.item())

        # early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        early_stopping(monitor_loss, model, epoch)
        best_val_loss = - early_stopping.best_score
        if early_stopping.early_stop:
            print("Early stopping")
            break

        # add pruning mechanism
        if not trial is None:
            trial.report(monitor_loss, epoch)

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        
        text = "Train Epoch [{}/{}] Loss: {:.5f}, Metric: {:.5f}".format(epoch, epochs, running_loss / (batch_idx+1), best_val_loss)
        train_bar.set_description(text)
    
    best_model = early_stopping.best_model
    print('Best model from epoch', early_stopping.best_epoch)
    results_dict = {
        'conf' : None,
        'loss_val' : np.array(loss_train_track),
        'loss_train' : np.array(loss_val_track),
        'metrics' : metrics_track
    }
    return best_model, results_dict