from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *
from model.data_preprocessing import load_and_preprocess

def run_model(data_path):

    X_train, X_test, y_train, y_test, _ = load_and_preprocess(data_path)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]

    results = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_prob),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred),
        "Confusion Matrix": confusion_matrix(y_test, y_pred)
    }

    return results
