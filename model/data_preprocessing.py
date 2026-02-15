import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess(data_path):

    df = pd.read_csv(data_path)

    df = df.drop(columns=["id"], errors="ignore")
    df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

    X = df.drop("diagnosis", axis=1)
    y = df["diagnosis"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler
