import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm, trange
import optuna


def predict_loop(dataloader, model):

    preds = []
    with torch.no_grad():
        for X in dataloader:
            X = X.cuda(non_blocking=True)
            pred = model(X)
            if X.shape[0] == 1:
                pred = torch.Tensor([pred]).cuda()
            preds.append(pred)
    preds = torch.cat(preds)
    return preds


def test_loop(dataloader, model, loss_fn, verbose):
    num_batches = len(dataloader)
    test_loss = 0

    with torch.no_grad():
        for X, z in dataloader:
            # X, z = X.cuda(non_blocking=True), z.cuda(non_blocking=True)
            pred = model(X)
            test_loss = test_loss + loss_fn(pred, z).item()

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
        retain_graph=True,
        create_graph=True,
    )[0]
    return dz_dxy


def pick_loss(name):
    if name == "l2":
        return nn.MSELoss()
    elif name == "l1":
        return nn.L1Loss()
    elif name == "huber":
        return nn.HuberLoss()


def estimate_density(
    dataset,
    dataset_test,
    model,
    opt,
    trial=None,
    return_model=True,
):

    name = opt.name + ".pth"
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=opt.lr,
        weight_decay=opt.wd,
    )

    L1_time_discrete = opt.L1_time_discrete
    if L1_time_discrete:
        lambda_t_d = opt.lambda_discrete
        loss_fn_t = nn.L1Loss()
    L1_time_gradient = opt.L1_time_gradient
    if L1_time_gradient:
        lambda_t_grad = opt.lambda_discrete
        # loss_fn_grad = nn.L1Loss()
    tvn = opt.tvn
    if tvn:
        std_data = torch.std(dataset.samples[:, 0:2], dim=0)
        mean_rd = torch.zeros((opt.bs, 2), device="cuda")
        std_rd = std_data * torch.ones((opt.bs, 2), device="cuda")
        tv_zeros = torch.zeros((opt.bs, 2), device="cuda")
        lambda_t_grad = opt.lambda_tvn
        loss_fn_tvn = pick_loss(opt.loss_tvn)

    loss_fn = nn.MSELoss()

    model.train()
    best_test_score = np.inf
    best_epoch = 0
    if opt.p.verbose:
        e_iterator = trange(1, opt.p.epochs + 1)
    else:
        e_iterator = range(1, opt.p.epochs + 1)

    for epoch in e_iterator:
        running_loss, total_num = 0.0, 0
        n_data = len(dataset)
        batch_idx = torch.randperm(n_data).cuda()
        bs = opt.bs
        if opt.p.verbose:
            train_iterator = tqdm(range(0, n_data, bs))
        else:
            train_iterator = range(0, n_data, bs)
        for i in train_iterator:
            idx = batch_idx[i:(i + bs)]
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                target_pred = model(dataset.samples[idx])
                lmse = loss_fn(target_pred, dataset.targets[idx])

                if L1_time_discrete:
                    t_t = model(dataset.samples_t[idx])
                    loss = lmse + lambda_t_d * loss_fn_t(target_pred, t_t)
                else:
                    loss = lmse

                if tvn:
                    ind = torch.randint(
                        0,
                        n_data,
                        size=(bs,),
                        requires_grad=False,
                        device="cuda",
                    )
                    noise = torch.normal(mean_rd, std_rd)
                    x_sample = dataset.samples[ind, :]
                    # from torchviz import make_dot
                    # import pdb; pdb.set_trace()
                    x_sample[:, 0:2] += noise
                    dz_dxy = continuous_diff(x_sample, model)
                    tv_norm = loss_fn_tvn(dz_dxy[:, 0:2], tv_zeros)
                    loss = loss + lambda_t_grad * tv_norm

            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()

            if opt.p.verbose:
                running_loss = running_loss + lmse.item()
                total_num = total_num + 1
                text = "Train Epoch [{}/{}] Loss: {:.4f}".format(
                    epoch, opt.p.epochs, running_loss / total_num
                )
                train_iterator.set_description(text)

        if epoch == 1:
            if return_model:
                torch.save(model.state_dict(), name)
        if epoch % 10 == 0:
            test_score = test_loop(dataset_test, model, loss_fn, opt.p.verbose)
            if test_score < best_test_score:
                best_test_score = test_score
                best_epoch = epoch
                if opt.p.verbose:
                    print(f"best model is now from epoch {epoch}")
                if return_model:
                    torch.save(model.state_dict(), name)
            elif epoch - best_epoch > 20:
                for g in optimizer.param_groups:
                    g["lr"] = g["lr"] / 10
        if torch.isnan(loss):
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
