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


# def generate_batch(n, batch_size):
#     """Yields bacth of specified size"""
#     batch_idx = torch.randperm(n).cuda()

#     for i in trange(0, n, batch_size):
#         yield batch_idx[i : i + batch_size]


def estimate_density(
    dataset,
    dataset_test,
    model,
    hp,
    name,
    lambda_t=1.0,
    method="None",
    trial=None,
    return_model=True,
    verbose=True,
):
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=hp["lr"],
        weight_decay=hp["wd"],
    )
    # scaler = GradScaler()
    loss_fn = nn.MSELoss()
    L1_diff = method == "L1_diff"
    if L1_diff:
        loss_fn_t = nn.L1Loss()
    model.train()
    best_test_score = np.inf
    best_epoch = 0
    if verbose:
        e_iterator = trange(1, hp["epoch"] + 1)
    else:
        e_iterator = range(1, hp["epoch"] + 1)

    for epoch in e_iterator:
        # train_iterator = tqdm(dataset) if verbose else dataset
        running_loss, total_num = 0.0, 0
        n_data = len(dataset)
        batch_idx = torch.randperm(n_data).cuda()
        bs = hp["bs"]
        train_iterator = tqdm(range(0, n_data, bs))
        for i in train_iterator:
            idx = batch_idx[i:(i + bs)]
            # for data_tuple in train_iterator:
            # for idx in generate_batch(, ):
            optimizer.zero_grad()
            # if L1_diff:
            #     inp, inp_t, target = data_tuple
            # else:
            #     inp, target = data_tuple

            # inp, target = inp.cuda(non_blocking=True), target.cuda(
            #     non_blocking=True
            # )
            with torch.cuda.amp.autocast():
                target_pred = model(dataset.samples[idx])
                lmse = loss_fn(target_pred, dataset.targets[idx])

                if L1_diff:
                    # inp_t = inp_t.cuda(non_blocking=True)
                    t_t = model(dataset.samples_t[idx])
                    loss = lmse + lambda_t * loss_fn_t(target_pred, t_t)
                else:
                    loss = lmse
            loss.backward()
            optimizer.step()
            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()

            if verbose:
                running_loss = running_loss + loss.item()
                total_num = total_num + 1
                text = "Train Epoch [{}/{}] Loss: {:.4f}".format(
                    epoch, hp["epoch"], running_loss / total_num
                )
                train_iterator.set_description(text)

        if epoch % 10 == 0:
            test_score = test_loop(dataset_test, model, loss_fn, verbose)
            if test_score < best_test_score:
                best_test_score = test_score
                best_epoch = epoch
                if verbose:
                    print(f"best model is now from epoch {epoch}")
                if return_model:
                    torch.save(model.state_dict(), name)
            elif epoch - best_epoch > 20:
                for g in optimizer.param_groups:
                    g["lr"] = g["lr"] / 10
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
