import optuna
from functools import partial
from main import train_and_test, pred_test_save
from parser import parser_f


def add_config_optuna_to_opt(opt, trial):
    # p = read_yaml(opt.yaml_file)
    opt.lr = trial.suggest_float(
        "learning_rate",
        opt.p.lr[0],
        opt.p.lr[1],
        log=True,
    )
    opt.wd = trial.suggest_float(
        "weight_decay",
        opt.p.wd[0],
        opt.p.wd[1],
        log=True,
    )
    bs_int = trial.suggest_int(
        "bs_int",
        opt.p.bs[0],
        opt.p.bs[1],
    )
    opt.bs = int(2**bs_int)
    if opt.fourier:
        scale_int = trial.suggest_int(
            "scale_int",
            opt.p.scale[0],
            opt.p.scale[1],
        )
        opt.scale = int(2**scale_int)
        mapping_size_int = trial.suggest_int(
            "mapping_size_int",
            opt.p.mapping_size[0],
            opt.p.mapping_size[1],
        )
        opt.mapping_size = int(2**mapping_size_int)
    else:
        opt.scale = None
        opt.mapping_size = None

    opt.L1_time_discrete = "L1TD" in opt.method
    opt.L1_time_gradient = "L1TG" in opt.method
    opt.tvn = "TVN" in opt.method

    if opt.L1_time_discrete:
        opt.lambda_discrete = trial.suggest_float(
            "lambda_discrete",
            opt.p.lambda_discrete[0],
            opt.p.lambda_discrete[1],
            log=True,
        )
    else:
        opt.lambda_discrete = None
    if opt.L1_time_gradient:
        opt.lambda_gradient_time = trial.suggest_float(
            "lambda_gradient_time",
            opt.p.lambda_gradient_time[0],
            opt.p.lambda_gradient_time[1],
            log=True,
        )
    else:
        opt.lambda_gradient_time = None
    if opt.tvn:
        opt.lambda_tvn = trial.suggest_float(
            "lambda_tvn", opt.p.lambda_tvn[0], opt.p.lambda_tvn[1], log=True
        )
        opt.loss_tvn = trial.suggest_categorical("loss_tvn", opt.p.loss_tvn)
    else:
        opt.lambda_tvn = None
        opt.loss_tvn = None

    opt.activation = trial.suggest_categorical("act", opt.p.act)
    opt.architecture = trial.suggest_categorical("architecture", opt.p.arch)
    return opt


def objective(opt, trial):

    opt = add_config_optuna_to_opt(opt, trial)
    print(opt)
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

    opt.L1_time_discrete = "L1TD" in opt.method
    opt.L1_time_gradient = "L1TG" in opt.method
    opt.tvn = "TVN" in opt.method

    if opt.L1_time_discrete:
        opt.lambda_discrete = params["lambda_discrete"]
    else:
        opt.lambda_discrete = None
    if opt.L1_time_gradient:
        opt.lambda_gradient_time = params["lambda_gradient_time"]
    else:
        opt.lambda_gradient_time = None
    if opt.tvn:
        opt.lambda_tvn = params["lambda_tvn"]
        opt.loss_tvn = params["loss_tvn"]
    else:
        opt.lambda_tvn = None
        opt.loss_tvn = None

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
    study.optimize(obj, n_trials=options.p.trials)

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
