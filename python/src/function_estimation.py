import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm, trange
import optuna
from torchlars import LARS


class EarlyStopper:
    def __init__(self, patience=1, testing_epoch=5, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf
        self.testing_epoch = testing_epoch

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss - self.min_delta:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += self.testing_epoch
            if self.counter >= self.patience:
                return True
        return False


def predict_loop(dataset, bs, model):
    n_data = len(dataset)
    batch_idx = torch.arange(0, n_data, dtype=int, device="cuda")
    train_iterator = tqdm(range(0, n_data, bs))
    preds = []
    with torch.no_grad():
        for i in train_iterator:
            idx = batch_idx[i : (i + bs)]
            pred = model(dataset.samples[idx])
            preds.append(pred)
    preds = torch.cat(preds)
    return preds


def test_loop(dataset, model, bs, loss_fn, verbose):
    n_data = len(dataset)
    num_batches = n_data // bs
    batch_idx = torch.arange(0, n_data, dtype=int, device="cuda")
    test_loss = 0
    if verbose:
        train_iterator = tqdm(range(0, n_data, bs))
    else:
        train_iterator = range(0, n_data, bs)
    with torch.no_grad():
        for i in train_iterator:
            idx = batch_idx[i : (i + bs)]
            pred = model(dataset.samples[idx])
            test_loss = test_loss + loss_fn(pred, dataset.targets[idx]).item()

    test_loss /= num_batches
    if verbose:
        print(f"Test Error: Avg loss: {test_loss:>8f}")
    return test_loss


def continuous_diff(x, model):
    torch.set_grad_enabled(True)
    x.requires_grad_(True)
    # x in [N,nvarin]
    # y in [N,nvarout]
    y = model(x)
    # dy in [N,nvarout]
    dz_dxy = torch.autograd.grad(
        y,
        x,
        torch.ones_like(y),
        create_graph=True,
    )[0]
    return dz_dxy


def estimate_density(
    dataset,
    dataset_test,
    model,
    opt,
    name,
    trial=None,
    return_model=True,
):

    name = name + ".pth"

    early_stopper = EarlyStopper(patience=15)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=opt.lr,
    )
    optimizer = LARS(optimizer=optimizer, eps=1e-8, trust_coef=0.001)

    td = opt.TD
    if td:
        lambda_td = opt.lambda_td
        loss_fn_t = nn.L1Loss()
    tvn = opt.tvn
    if tvn:
        std_data = torch.std(dataset.samples[:, 0:2], dim=0)
        mean_xy = torch.zeros((opt.bs, 2), device="cuda")
        std_xy = std_data * torch.ones((opt.bs, 2), device="cuda")
        lambda_tvn = opt.lambda_tvn
        loss_tvn = nn.L1Loss()
    cont_grad = tvn

    loss_fn = nn.MSELoss()

    model.train()
    best_test_score = np.inf
    best_epoch = 0
    if opt.verbose:
        e_iterator = trange(1, opt.epochs + 1)
    else:
        e_iterator = range(1, opt.epochs + 1)

    for epoch in e_iterator:
        running_loss, total_num = 0.0, 0
        n_data = len(dataset)
        batch_idx = torch.randperm(n_data).cuda()
        bs = opt.bs
        if opt.verbose:
            train_iterator = tqdm(range(0, n_data, bs))
        else:
            train_iterator = range(0, n_data, bs)

        for i in train_iterator:
            idx = batch_idx[i : (i + bs)]
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                target_pred = model(dataset.samples[idx])
                lmse = loss_fn(target_pred, dataset.targets[idx])

                if td:
                    t_t = model(dataset.samples_t[idx])
                    loss = lmse + lambda_td * loss_fn_t(target_pred, t_t)
                else:
                    loss = lmse

                if cont_grad:
                    ind = torch.randint(
                        0,
                        n_data,
                        size=(bs,),
                        requires_grad=False,
                        device="cuda",
                    )
                    x_sample = dataset.samples[ind, :]
                    x_sample.requires_grad_(False)
                    if tvn:
                        noise_xy = torch.normal(mean_xy, std_xy)
                        x_sample[:, 0:2] += noise_xy
                    dz_dxy = continuous_diff(torch.Tensor(x_sample), model)
                    if tvn:
                        loss = loss + lambda_tvn * loss_tvn(dz_dxy[:, 0:2], mean_xy)

            loss.backward()
            optimizer.step()

            if opt.verbose:
                running_loss = running_loss + lmse.item()
                total_num = total_num + 1
                text = "Train Epoch [{}/{}] Loss: {:.4f}".format(
                    epoch, opt.epochs, running_loss / total_num
                )
                train_iterator.set_description(text)

        if epoch == 1:
            if return_model:
                torch.save(model.state_dict(), name)
        if epoch % 5 == 0:
            test_score = test_loop(
                dataset_test,
                model,
                opt.bs,
                loss_fn,
                opt.verbose,
            )
            if test_score < best_test_score:
                best_test_score = test_score
                best_epoch = epoch
                if opt.verbose:
                    print(f"best model is now from epoch {epoch}")
                if return_model:
                    torch.save(model.state_dict(), name)
            if epoch - best_epoch > 10:
                for g in optimizer.param_groups:
                    g["lr"] = g["lr"] / 10
            if early_stopper.early_stop(test_score):
                break

        if not torch.isfinite(loss):
            break
        # Add prune mechanism
        if trial:
            trial.report(lmse, epoch)

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    if return_model:
        model.load_state_dict(torch.load(name))
        return model, best_test_score
    else:
        return best_test_score
