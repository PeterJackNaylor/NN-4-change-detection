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


def test_loop(dataloader, model, loss_fn):
    num_batches = len(dataloader)
    test_loss = 0

    with torch.no_grad():
        for X, z in dataloader:
            X, z = X.cuda(non_blocking=True), z.cuda(non_blocking=True)
            pred = model(X)
            test_loss = test_loss + loss_fn(pred, z).item()

    test_loss /= num_batches

    print(f"\n Test Error: \n Avg loss: {test_loss:>8f} \n")
    return test_loss


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
    for epoch in trange(1, hp["epoch"] + 1):

        running_loss, total_num, train_bar = 0.0, 0, tqdm(dataset)
        for data_tuple in train_bar:
            # with torch.autograd.set_detect_anomaly(True):
            # with autocast(device_type='cuda', dtype=torch.float16):

            optimizer.zero_grad()
            if L1_diff:
                inp, inp_t, target = data_tuple
            else:
                inp, target = data_tuple

            # inp, target = inp.cuda(non_blocking=True), target.cuda(
            #     non_blocking=True
            # )
            target_pred = model(inp)
            lmse = loss_fn(target_pred, target)

            if L1_diff:
                # inp_t = inp_t.cuda(non_blocking=True)
                target_pred_t = model(inp_t)
                loss = lmse + lambda_t * loss_fn_t(target_pred, target_pred_t)
            else:
                loss = lmse
            loss.backward()
            optimizer.step()
            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()
            # opti.step()
            running_loss = running_loss + loss.item()
            total_num = total_num + 1
            text = "Train Epoch [{}/{}] Loss: {:.4f}".format(
                epoch, hp["epoch"], running_loss / total_num
            )
            train_bar.set_description(text)

        if epoch % 10 == 0:
            test_score = test_loop(dataset_test, model, loss_fn)
            if test_score < best_test_score:
                best_test_score = test_score
                best_epoch = epoch
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
