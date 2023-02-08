import optuna
from functools import partial
from main import train_and_test, pred_test_save
from parser import read_yaml, parser_f


def objective(opt, trial):
    p = read_yaml(opt.yaml_file)
    opt.lr = trial.suggest_float(
        "learning_rate",
        p["lr"][0],
        p["lr"][1],
        log=True,
    )
    opt.wd = trial.suggest_float(
        "weight_decay",
        p["wd"][0],
        p["wd"][1],
        log=True,
    )
    bs_int = trial.suggest_int(
        "bs_int",
        p["bs"][0],
        p["bs"][1],
    )
    opt.bs = int(2**bs_int)
    if opt.fourier:
        scale_int = trial.suggest_int(
            "scale_int",
            p["scale"][0],
            p["scale"][1],
        )
        opt.scale = int(2**scale_int)
        mapping_size_int = trial.suggest_int(
            "mapping_size_int",
            p["mp_size"][0],
            p["mp_size"][1],
        )
        opt.mapping_size = int(2**mapping_size_int)
    else:
        opt.scale = None
        opt.mapping_size = None
    if opt.method == "L1_diff":
        opt.lambda_t = trial.suggest_float(
            "lambda_t", p["lambda_t"][0], p["lambda_t"][1], log=True
        )
    else:
        opt.lambda_t = None
    opt.activation = trial.suggest_categorical("act", p["act"])
    opt.architecture = trial.suggest_categorical("architecture", p["arch"])

    time = 0 if opt.csv1 else -1

    best_score = train_and_test(time, opt, trial=trial, return_model=False)
    return best_score


def return_best_model(opt, params):
    opt.lr = params["learning_rate"]
    opt.wd = params["weight_decay"]
    bs_int = params["bs_int"]
    opt.bs = int(2**bs_int)
    if opt.fourier:
        scale_int = params["scale_int"]
        opt.scale = int(2**scale_int)
        mapping_size_int = params["mapping_size_int"]
        opt.mapping_size = int(2**mapping_size_int)
    opt.architecture = params["architecture"]
    if opt.method == "L1_diff":
        opt.lambda_t = params["lambda_t"]
    else:
        opt.lambda_t = None
    opt.activation = params["act"]

    time = 0 if opt.csv1 else -1

    model, B, nv, best_score = train_and_test(time, opt, return_model=True)
    return model, B, nv, best_score, opt


def main():
    options = parser_f()
    study = optuna.create_study(
        study_name=options.name,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.HyperbandPruner(),
    )
    obj = partial(objective, options)
    study.optimize(obj, n_trials=options.trials)

    best_params = study.best_trial.params

    model, B, nv, best_score, opt = return_best_model(options, best_params)
    time = 0 if options.csv1 else -1
    pred_test_save(B, nv, time, model, best_score, opt)

    for key, value in best_params.items():
        print("{}: {}".format(key, value))

    fig = optuna.visualization.plot_intermediate_values(study)
    fig.write_image(options.name + "_inter_optuna.png")

    fig = optuna.visualization.plot_parallel_coordinate(study)
    fig.write_image(options.name + "_searchplane.png")

    fig = optuna.visualization.plot_param_importances(study)
    fig.write_image(options.name + "_important_params.png")


if __name__ == "__main__":
    main()
