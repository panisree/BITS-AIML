import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
def load_and_preprocess(data_path):

    if isinstance(data_path, pd.DataFrame):
        df = data_path.copy()
    else:
        df = pd.read_csv(data_path)

    df = df.drop(columns=["id"], errors="ignore")

    # Clean diagnosis column
    df["diagnosis"] = df["diagnosis"].astype(str).str.strip()

    if set(df["diagnosis"].unique()) <= {"M", "B"}:
        df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})
    else:
        df["diagnosis"] = pd.to_numeric(df["diagnosis"], errors="coerce")

    # Drop invalid rows
    df = df.dropna(subset=["diagnosis"])

    X = df.drop("diagnosis", axis=1)
    y = df["diagnosis"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler
