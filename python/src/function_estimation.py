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
        # retain_graph=True,
        create_graph=True,
    )[0]
    return dz_dxy


def pick_loss(name):
    if name == "l2":
        return nn.MSELoss(reduction="none")
    elif name == "l1":
        return nn.L1Loss(reduction="none")
    elif name == "huber":
        return nn.HuberLoss(reduction="none")


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
        weight_decay=opt.wd,
    )
    optimizer = LARS(optimizer=optimizer, eps=1e-8, trust_coef=0.001)

    L1_time_discrete = opt.L1_time_discrete
    if L1_time_discrete:
        print("Using L1TD")
        lambda_t_d = opt.lambda_discrete
        loss_fn_t = nn.L1Loss()
    L1_time_gradient = opt.L1_time_gradient
    if L1_time_gradient:
        print("Using L1TG")
        lambda_tvn_t = opt.lambda_tvn_t
        lambda_tvn_t_sd = opt.lambda_tvn_t_sd
        mean_t = torch.zeros((opt.bs,), device="cuda")
        std_t = lambda_tvn_t_sd * torch.ones((opt.bs,), device="cuda")
    else:
        lambda_tvn_t = 0
    tvn = opt.tvn
    if tvn:
        print("Using TVN")
        std_data = torch.std(dataset.samples[:, 0:2], dim=0)
        mean_xy = torch.zeros((opt.bs, 2), device="cuda")
        std_xy = std_data * torch.ones((opt.bs, 2), device="cuda")
        lambda_tvn = opt.lambda_tvn
    else:
        lambda_tvn = 0
    cont_grad = L1_time_gradient or tvn
    if cont_grad:
        coef = torch.Tensor([lambda_tvn, lambda_tvn, lambda_tvn_t]).to("cuda")
        grad_zeros = torch.zeros((opt.bs, 3), device="cuda")
        loss_fn_grad = pick_loss(opt.loss_tvn)

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

                if L1_time_discrete:
                    t_t = model(dataset.samples_t[idx])
                    loss = lmse + lambda_t_d * loss_fn_t(target_pred, t_t)
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
                    if L1_time_gradient:
                        noise_t = torch.normal(mean_t, std_t)
                        x_sample[:, 2] += noise_t

                    dz_dxy = continuous_diff(torch.Tensor(x_sample), model)
                    tv_norm = (
                        coef * loss_fn_grad(dz_dxy, grad_zeros).sum(axis=0)
                    ).sum()
                    loss = loss + tv_norm

            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
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
