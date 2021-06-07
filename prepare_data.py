import os

import click
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

PATH_DATA = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data"
)


@click.command()
def prepare_data():
    data = load_iris(as_frame=True)
    df = data.frame
    df = df.rename(columns={
        "sepal length (cm)": "sepal length",
        "sepal width (cm)": "sepal width",
        "petal length (cm)": "petal length",
        "petal width (cm)": "petal width",
        "target": "species",
    })
    df["species"] = df["species"].replace({
        0: "setosa",
        1: "versicolor",
        2: "virginica"
    })

    # 1- split into private train and test sets.
    random_state = 57
    df_train, df_test = train_test_split(
        df, test_size=0.2, random_state=random_state,
    )
    # 2- the private train set is also the public set which we need to split
    # into a train and test set.
    df_public_train, df_public_test = train_test_split(
        df_train, test_size=0.2, random_state=random_state,
    )
    # 3- save all the data
    if not os.path.exists(PATH_DATA):
        os.makedirs(PATH_DATA)
    df_train.to_csv(
        os.path.join(PATH_DATA, "train.csv"),
        index=False,
    )
    df_test.to_csv(
        os.path.join(PATH_DATA, "test.csv"),
        index=False,
    )
    public_dir = os.path.join(PATH_DATA, "public")
    if not os.path.exists(public_dir):
        os.makedirs(public_dir)
    df_public_train.to_csv(
        os.path.join(public_dir, "train.csv"),
        index=False,
    )
    df_public_test.to_csv(
        os.path.join(public_dir, "test.csv"),
        index=False,
    )


if __name__ == "__main__":
    prepare_data()
