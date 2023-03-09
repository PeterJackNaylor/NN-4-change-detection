from boxplots import parser_f, prep_data
import plotly.graph_objects as go
from scipy import signal


def naming_scheme(fs, meth, data):
    if fs == "FourierBasis":
        sfs = "FB"
    elif fs == "SIREN":
        sfs = "S"
    if meth == "single_M+TVN":
        smeth = "TVN"
    elif meth == "single_M+TD":
        smeth = "TD"
    else:
        smeth = meth
        new_name = f"{data} - {sfs}"
        return new_name
    new_name = f"{data} - {sfs} + {smeth}"
    return new_name


new_name_mapping = {
    "(3) - S + TVN": ["SIREN + TVN", "#a6cee3"],
    "(3) - S + TD": ["SIREN + TD", "#b2df8a"],
    "(3) - FB + TVN": ["RFF + TVN", "#1f78b4"],
    "(3) - FB + TD": ["RFF + TD", "#33a02c"],
    "(3) - FB": ["RFF", "#e31a1c"],
    "(3) - S": ["SIREN", "#fb9a99"],
}


def plot(df, baseline, var):
    df = df.copy()
    # df["data"] = "All"
    # df = df[df["data"] == "(5)"]
    df["new_name"] = df.apply(
        lambda row: naming_scheme(row["fs"], row["method"], row["data"]), axis=1
    )

    df_mean = df.groupby(["lambda_t", "new_name"]).mean().reset_index()
    df_std = df.groupby(["lambda_t", "new_name"]).std().reset_index()
    size = df.groupby(["lambda_t", "new_name"]).size().reset_index()
    layout = go.Layout(autosize=True, margin={"l": 0, "r": 0, "t": 20, "b": 0})

    fig = go.Figure(layout=layout)
    uniques = df["new_name"].unique()
    uniques.sort()

    for name in uniques:
        # fig = px.line(df, x="lambda_t", y=var, color="new_name", log_x=True)
        mean_name = df_mean.copy()[df_mean["new_name"] == name]
        std_name = df_std.copy()[df_mean["new_name"] == name]
        size_name = size.copy()[df_mean["new_name"] == name]
        x = mean_name["lambda_t"].values
        y = mean_name[var].values
        y = signal.savgol_filter(y, 5, 3)
        std = std_name[var].values
        n_sqrt = size_name[0].values ** 0.5
        error = 1.96 * std / n_sqrt
        error = signal.savgol_filter(error, 5, 3)
        nname, color = new_name_mapping[name]
        scat = go.Scatter(
            x=x,
            y=y,
            error_y=dict(
                type="data",  # value of error bar given in data coordinates
                array=error,
                visible=True,
            ),
            name=nname,
            line=dict(color=color),
        )
        fig.add_trace(scat)
    baseline["new_name"] = baseline.apply(
        lambda row: naming_scheme(row["fs"], "", row["data"]), axis=1
    )
    baseline_mean = baseline.groupby(["lambda_t", "new_name"]).mean().reset_index()
    baseline_std = baseline.groupby(["lambda_t", "new_name"]).std().reset_index()
    baselinesize = baseline.groupby(["lambda_t", "new_name"]).size().reset_index()
    uniques = baseline["new_name"].unique()
    uniques.sort()
    for name in uniques:
        vline_mean = baseline_mean[baseline_mean["new_name"] == name][var].values[0]
        vline_std = baseline_std[baseline_mean["new_name"] == name][var].values[0]
        vline_size = baselinesize[baseline_mean["new_name"] == name][0].values[0]
        n_sqrt = vline_size**0.5
        error = 1.96 * vline_std / n_sqrt
        y = [vline_mean for lt in x]
        error = [vline_std for lt in x]
        nname, color = new_name_mapping[name]
        scat = go.Scatter(
            x=x,
            y=y,
            error_y=dict(
                type="data",  # value of error bar given in data coordinates
                array=error,
                visible=True,
            ),
            name=nname,
            line=dict(color=color),
        )
        # fig.add_trace(scat)

        scat = go.Scatter(
            x=x,
            y=y,
            mode="lines",
            name=nname,
            line=dict(color=color, width=4, dash="dash"),
        )
        fig.add_trace(scat)

    fig.update_xaxes(type="log", title="&#955;")
    fig.update_yaxes(title=var)
    fig.update_layout(
        font=dict(
            size=18,
        )
    )
    return fig


def main():
    opt = parser_f()
    df = prep_data(opt.csv)
    df["MAE"] = (df["MAE_PC0"] + df["MAE_PC1"]) / 2
    baseline = df.copy()[df["lambda_t"] == 0]
    baseline["RMSE"] = baseline["MSE"] ** 0.5
    baseline["IoU"] = baseline["IoU_gmm"]
    df["RMSE"] = df["MSE"] ** 0.5
    df["IoU"] = df["IoU_gmm"]

    fig_iou = plot(df, baseline, "IoU")
    fig_iou.update_layout(showlegend=False)
    fig_iou.write_image("IoU.png")

    fig_mse = plot(df, baseline, "RMSE")
    fig_mse.update_yaxes(range=[3, 5])
    fig_mse.write_image("MSE.png")

    fig_mae = plot(df, baseline, "MAE")
    fig_mae.update_yaxes(range=[1.8, 3])
    fig_mae.write_image("MAE.png")
    # plot(df, "MSE_PC0")
    # plot(df, "MSE_PC1")


if __name__ == "__main__":
    main()
