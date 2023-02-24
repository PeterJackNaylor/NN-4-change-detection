import numpy as np
import pandas as pd
import argparse
import plotly.express as px
import plotly.graph_objects as go


def parser_f():

    parser = argparse.ArgumentParser(
        description="Train supervised NN on cell",
    )
    parser.add_argument(
        "--csv",
        type=str,
    )
    args = parser.parse_args()

    return args


def data_branch(name):
    name = name.replace("two", "")
    name = name.replace("one", "")
    name = name.replace("zero", "")
    name = name[:-1]
    return name


def give_name(row):
    if row["Fourier"]:
        return "{} ({})".format(row["data"], row["method"])
    else:
        return


name_mapping = {
    "LyonS": "(2)",
    "Lidar3": "(3)",
    "MultiSensor": "(5)",
    "Photogrametry": "(4)",
    "Lidar05": "(1)",
}


def prep_data(csv):
    df = pd.read_csv(csv)
    # define dictionary
    name_dic = {"Unnamed: 0": "dataset"}
    df = df.rename(columns=name_dic)

    df["data"] = df["dataset"].apply(lambda row: data_branch(row))
    df["data"] = df["data"].apply(lambda row: name_mapping[row])
    df["name"] = df.apply(
        lambda row: "{} ({})".format(row["data"], row["method"]), axis=1
    )
    auc_var = ["AUC_addition", "AUC_deletion", "AUC_nochange"]
    df["AUC"] = df[auc_var].mean(axis=1)
    df["MSE"] = (df["MSE_PC0"] + df["MSE_PC1"]) / 2
    return df


def box_plot_fourrier(table, var):
    t = table.copy()
    t = t[t["method"] == "single_M+L1TD"]

    fig = px.box(t, x="name", y=var, color="fs")
    fig.write_image("box_plot_importance_fourrier_{}.png".format(var))


def box_plot_different_method(table, var, only_good=True):
    t = table.copy()
    t = t[t["fs"] == "FourierBasis"]
    if only_good:
        suffixe = "only_good"
        t = t[~t["method"].isin(["double_M", "double_M+TVN"])]
    else:
        suffixe = ""
    METHODS = list(t.method.unique())

    fig = go.Figure()
    for meth in METHODS:
        tmp = t.copy()
        tmp = tmp[tmp["method"] == meth]
        fig.add_trace(
            go.Box(
                y=tmp[var],
                x=tmp["data"],
                boxmean=True,  # represent mean
                name=meth,
            )
        )
    if var == "IoU_gmm":
        nice_var = "IoU"
    else:
        nice_var = var
    fig.update_layout(
        yaxis_title="{}".format(nice_var),
        boxmode="group",
    )
    fig.write_image("box_plot_method_{}{}.png".format(var, suffixe))


def table_avg(df):

    for var in df.fs.unique():
        table = df.copy()[df["fs"] == var].copy()
        auc = pd.pivot_table(
            table, values="AUC", index="data", columns="method", aggfunc=np.mean
        )
        auc.loc["Avg"] = auc.mean(axis=0)
        iou = pd.pivot_table(
            table,
            values="IoU_gmm",
            index="data",
            columns="method",
            aggfunc=np.mean,
        )
        iou.loc["Avg"] = iou.mean(axis=0)
        print("AUC {}".format(var))
        print(auc)
        print("IoU {}".format(var))
        print(iou)


def main():
    opt = parser_f()
    df = prep_data(opt.csv)
    # rename columns in DataFrame using dictionary
    box_plot_fourrier(df, "AUC")
    box_plot_fourrier(df, "IoU_gmm")

    box_plot_different_method(df, "IoU_gmm", only_good=True)
    box_plot_different_method(df, "AUC", only_good=True)
    box_plot_different_method(df, "MSE", only_good=True)
    box_plot_different_method(df, "IoU_gmm", only_good=False)
    box_plot_different_method(df, "AUC", only_good=False)
    box_plot_different_method(df, "MSE", only_good=False)

    table_avg(df)


if __name__ == "__main__":
    main()
