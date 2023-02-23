import optuna
from functools import partial
from main import train_and_test, pred_test_save
from parser import parser_f, AttrDict


def add_config_optuna_to_opt(opt, trial):
    model_hp = AttrDict()
    model_hp.fourier = opt.fourier
    model_hp.siren = opt.siren
    model_hp.verbose = opt.p.verbose
    model_hp.epochs = opt.p.epochs
    model_hp.lr = trial.suggest_float(
        "learning_rate",
        opt.p.lr[0],
        opt.p.lr[1],
        log=True,
    )
    bs_int = trial.suggest_int(
        "bs_int",
        opt.p.bs[0],
        opt.p.bs[1],
    )
    model_hp.bs = int(2**bs_int)
    if model_hp.fourier:
        mapping_size_int = trial.suggest_int(
            "mapping_size_int",
            opt.p.mapping_size[0],
            opt.p.mapping_size[1],
        )
        model_hp.mapping_size = int(2**mapping_size_int)

    if model_hp.fourier or model_hp.siren:
        scale_int = trial.suggest_int(
            "scale_int",
            opt.p.scale[0],
            opt.p.scale[1],
        )
        model_hp.scale = int(2**scale_int)

    model_hp.L1_time_discrete = "L1TD" in opt.method
    model_hp.L1_time_gradient = "L1TG" in opt.method
    model_hp.tvn = "TVN" in opt.method

    if model_hp.L1_time_discrete:
        model_hp.lambda_discrete = trial.suggest_float(
            "lambda_discrete",
            opt.p.lambda_discrete[0],
            opt.p.lambda_discrete[1],
            log=True,
        )

    if model_hp.L1_time_gradient:
        model_hp.lambda_tvn_t = trial.suggest_float(
            "lambda_tvn_t",
            opt.p.lambda_tvn_t[0],
            opt.p.lambda_tnv_t[1],
            log=True,
        )
        model_hp.lambda_tvn_t_sd = trial.suggest_float(
            "lambda_tvn_t_sd",
            opt.p.lambda_tvn_t_sd[0],
            opt.p.lambda_tvn_t_sd[1],
            log=True,
        )

    if model_hp.tvn:
        model_hp.lambda_tvn = trial.suggest_float(
            "lambda_tvn", opt.p.lambda_tvn[0], opt.p.lambda_tvn[1], log=True
        )

    if model_hp.siren:
        model_hp.architecture = "siren"
        model_hp.siren_hidden_num = trial.suggest_int(
            "siren_hidden_num",
            opt.p.siren.hidden_num[0],
            opt.p.siren.hidden_num[1],
        )
        siren_hidden_dim_int = trial.suggest_int(
            "siren_hidden_dim_int",
            opt.p.siren.hidden_dim[0],
            opt.p.siren.hidden_dim[1],
        )
        model_hp.siren_hidden_dim = int(2**siren_hidden_dim_int)
        siren_do_skip_int = trial.suggest_categorical(
            "do_skip",
            opt.p.siren.do_skip,
        )
        model_hp.siren_skip = siren_do_skip_int == 1

    else:
        model_hp.architecture = trial.suggest_categorical(
            "architecture",
            opt.p.arch,
        )
        model_hp.activation = trial.suggest_categorical("act", opt.p.act)
    return model_hp


def objective(opt, trial):

    model_hp = add_config_optuna_to_opt(opt, trial)
    time = 0 if opt.csv1 else -1

    model_hp = train_and_test(
        time,
        opt,
        model_hp,
        trial=trial,
        return_model=False,
    )
    return model_hp.best_score


def return_best_model(opt, params):

    model_hp = AttrDict()
    model_hp.fourier = opt.fourier
    model_hp.siren = opt.siren
    model_hp.verbose = opt.p.verbose
    model_hp.epochs = opt.p.epochs
    model_hp.lr = params["learning_rate"]
    bs_int = params["bs_int"]
    model_hp.bs = int(2**bs_int)
    if model_hp.siren or opt.fourier:
        scale_int = params["scale_int"]
        model_hp.scale = int(2**scale_int)
    if model_hp.fourier:
        mapping_size_int = params["mapping_size_int"]
        model_hp.mapping_size = int(2**mapping_size_int)
    if model_hp.siren:
        model_hp.architecture = "siren"
        model_hp.siren_hidden_num = params["siren_hidden_num"]
        siren_hidden_dim_int = params["siren_hidden_dim_int"]
        model_hp.siren_hidden_dim = int(2**siren_hidden_dim_int)
        model_hp.siren_skip = params["do_skip"] == 1
    else:
        model_hp.architecture = params["architecture"]
        model_hp.activation = params["act"]

    model_hp.L1_time_discrete = "L1TD" in opt.method
    model_hp.L1_time_gradient = "L1TG" in opt.method
    model_hp.tvn = "TVN" in opt.method

    if model_hp.L1_time_discrete:
        model_hp.lambda_discrete = params["lambda_discrete"]

    if model_hp.L1_time_gradient:
        model_hp.lambda_tvn_t = params["lambda_tvn_t"]
        model_hp.lambda_tvn_t_sd = params["lambda_tvn_t_sd"]

    if model_hp.tvn:
        model_hp.lambda_tvn = params["lambda_tvn"]

    time = 0 if opt.csv1 else -1
    model, model_hp = train_and_test(
        time,
        opt,
        model_hp,
        trial=None,
        return_model=True,
    )
    return model, model_hp


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

    model, model_hp = return_best_model(options, best_params)
    time = 0 if options.csv1 else -1
    pred_test_save(model, model_hp, time, options)

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
