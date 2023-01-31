import pandas as pd
from sklearn.preprocessing import minmax_scale, scale


def load_and_preprocess(
    onehot=True,
    rescale_features=False,
    rescale_target=False,
    drop_oneof_onehot=False,
    drop_X4=False,
):
    """
    Loads and preprocesses the data.
    Rescaling is only needed for SVM/SVR and LogReg/LinReg.

    :param onehot: do one-hot encoding of X2,X3 (bool)
    :param rescale_features: rescale X4,X5,X6 (bool) to mu=0.5, sigma**2=1
    :param rescale_target: rescale Y to range [0,1] (bool)
    :param drop_oneof_onehot: drop one of the one-hot encoded columns (bool)
    :param drop_X4: drop X4 (bool)
    :returns df:
    :rtype: pd.DataFrame
    """
    df = pd.read_csv("biological_data.csv", index_col=0)
    df.drop_duplicates(inplace=True)
    df.drop(labels=["X1"], axis=1, inplace=True)
    df.reset_index(drop=True, inplace=True)

    if onehot is True:
        # encode X2 and X3 to one hot
        df_dummies_X2 = pd.get_dummies(df.X2, prefix="X2")
        df_dummies_X3 = pd.get_dummies(df.X3, prefix="X3")
        df = df.drop(labels=["X2", "X3"], axis=1)
        df = df.merge(df_dummies_X2, left_index=True, right_index=True)
        df = df.merge(df_dummies_X3, left_index=True, right_index=True)

    if rescale_features is True:
        # center the gaussians (X4, X5, X6) at 0.5 and rescale to sigma**2 = 1
        # so they fit(-ish) between 0 and 1
        # +0.5 so they're in the same(-ish) range as the one-hot vectors
        # NB: doing this before train-test split can leak data
        #     i don't quite see how, but meh...
        df[["X4", "X5", "X6"]] = scale(df[["X4", "X5", "X6"]], axis=0) + 0.5

    # create new column as classification target
    df["Yc"] = df.Y > 50

    # rescale target to 0 to 1 (for SVM/SVR and log
    if rescale_target is True:
        df.Y = minmax_scale(df.Y)

    # drop the last of each one-hot encoding (X2 and X3)
    if drop_oneof_onehot is True:
        df = df.drop(
            labels=[
                "X2_XSHSMRYFDTAVSRPGRGEPRFISVGYVDDTQFVRFDSDAASPRGEPRAPWVEQEGPEYWDRETQKYKRQAQADRVNLRKLRGYYNQSED",
                "X3_XSHIIQRMYGCDLGPDGRLLRGHDQLAYDGKDYIALNEDLRSWTAADTAAQITQRKWEAAREAEQLRAYLEGTCVEWLRRYLENGKETLQRA",
            ],
            axis=1,
        )

    # drop X4 (which is highly correlated with X5)
    if drop_X4 is True:
        df = df.drop(labels=["X4"], axis=1)

    return df


def df_to_Xy(df, to_class=False):
    """
    Converts dataframe to X, y vectors for further consumption.

    :param to_class: returns class if True, float if False
    :returns X, y: feature and target vectors
    :rtyep tuple:
    """
    cols = list(df.columns)
    cols.remove("Y")
    cols.remove("Yc")
    X = df[cols]
    if to_class is True:
        y = df["Yc"]
    else:
        y = df["Y"]
    return X, y
