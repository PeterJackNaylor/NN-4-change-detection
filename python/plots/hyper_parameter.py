from boxplots import parser_f, prep_data
import plotly.express as px


def naming_scheme(fs, meth, data):
    if fs == "FourierBasis":
        sfs = "FB"
    elif fs == "SIREN":
        sfs = "S"
    if meth == "single_M+TVN":
        smeth = "TVN"
    elif meth == "single_M+L1TD":
        smeth = "L1TD"
    new_name = f"{data} - {sfs} + {smeth}"
    return new_name


def plot(df, var):
    df = df.copy()
    # df["data"] = "All"
    df = df[df["data"] == "(5)"]
    df["new_name"] = df.apply(
        lambda row: naming_scheme(row["fs"], row["method"], row["data"]), axis=1
    )
    df = df.groupby(["lambda_t", "new_name"]).mean().reset_index()
    fig = px.line(df, x="lambda_t", y=var, color="new_name", log_x=True)
    fig.show()


def main():
    opt = parser_f()
    df = prep_data(opt.csv)
    plot(df, "IoU_gmm")
    plot(df, "MSE")


if __name__ == "__main__":
    main()
