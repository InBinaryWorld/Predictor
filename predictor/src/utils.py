import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.base import clone
from sklearn.metrics import make_scorer, confusion_matrix
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (f1_score, precision_score,
                             recall_score, roc_auc_score)
from data import VERBOSE, SEED


def prepare_data(dir, delimiter=","):
    encoder = LabelEncoder()
    data = pd.read_csv(dir, delimiter=delimiter)
    data['r_Language'] = encoder.fit_transform(data['r_Language'])

    x = data.iloc[:, 4:]
    y = data.iloc[:, 3:4]

    x_v = x.values  # returns a numpy array
    min_max_scaler = MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x_v)
    x_df = pd.DataFrame(x_scaled)
    x_df.columns = x.columns

    return x_df, y, encoder


def prepare_and_split_data(dir, delimiter=","):
    x, y, encoder = prepare_data(dir, delimiter)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.4, random_state=SEED, stratify=y
    )
    return x_train, x_test, y_train, y_test, encoder


def test_classifier(name, x, y, clf, skf):
    scores, _ = classify(x, y, clf, skf)
    classify_report(name, scores)


def classify(x, y, clf, skf):
    x, y, labels = x.values, y.values, list(y.columns)

    fold_results = []
    for train, test in skf.split(x, y):
        clf_clone = clone(clf)
        x_train, x_test, y_train, y_test = x[train], x[test], y[train], y[test]

        clf_clone.fit(x_train, y_train)
        y_prob = clf_clone.predict_proba(x_test)
        y_pred = clf_clone.predict(x_test)

        (precision, recall, f1, mcc, auc) = calculate_metrics(y_test.T[0], y_pred, y_prob)

        fold_results.append({
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mcc": mcc,
            "auc": auc
        })

    scores = {}
    for key in fold_results[0].keys():
        scores[key] = np.mean([row[key] for row in fold_results])

    return scores, fold_results


def calculate_metrics(y_true, y_pred, y_proba):
    return (
        precision_score(y_true, y_pred, zero_division=0),
        recall_score(y_true, y_pred, zero_division=0),
        f1_score(y_true, y_pred, zero_division=0),
        matthews_corrcoef(y_true, y_pred),
        roc_auc_score(y_true, y_proba[:, [1]])
    )


def classify_report(name, scores):
    print(f"******** {name} ********")
    print("{: <15}{: <15}{: <15}{: <15}{: <15}".format(
        "Precision", "Recall", "F1", "MCC", "AUC"), flush=True)
    print("{:<15.3f}{:<15.3f}{:<15.3f}{:<15.3f}{:<15.3f}".format(
        scores["precision"], scores["recall"], scores["f1"], scores["mcc"], scores["auc"]), flush=True)


def feature_importance_rank(name, x, y, clf):
    clf_clone = clf
    results = clf_clone.fit(x, y)

    features = [{'name': x.columns[idx], 'value': value} for idx, value in enumerate(results.feature_importances_)]
    features.sort(key=lambda x: x["value"], reverse=True)

    print(f"******** {name} ********")
    for feature in features:
        print("{: <40}{: <20.2f}%".format(feature["name"], feature["value"] * 100), flush=True)


def precision_scorer():
    return make_scorer(precision_score, zero_division=0)


def recall_scorer():
    return make_scorer(recall_score, zero_division=0)


def f1_scorer():
    return make_scorer(f1_score, zero_division=0)


def mcc_scorer():
    return make_scorer(matthews_corrcoef)


def optimize_for_grid(name, x, y, clf, skf, grid, scorer):
    current_clf = clone(clf)
    print("Current Time =", datetime.now().strftime("%H:%M:%S"))
    grid_cv = GridSearchCV(current_clf, grid, scoring=scorer, n_jobs=-1, cv=skf, verbose=True)

    grid_cv.fit(x, y)
    print(f'******** {name} ********', flush=True)
    print(f'best score: {round(grid_cv.best_score_, 4)}')
    print(f'best param: {grid_cv.best_params_}')
    print(grid_cv.cv_results_)


def optimize_for_random(name, x, y, clf, skf, grid, scorer, iterations, seed):
    current_clf = clone(clf)
    print("Current Time =", datetime.now().strftime("%H:%M:%S"))
    random_cv = RandomizedSearchCV(current_clf, grid, scoring=scorer, n_jobs=-1, cv=skf,
                                   n_iter=iterations, random_state=seed, verbose=True)

    random_cv.fit(x, y)
    print(f'******** {name} ********', flush=True)
    print(f'best score: {round(random_cv.best_score_, 4)}')
    print(f'best param: {random_cv.best_params_}')
    print(random_cv.cv_results_)


def log(*args, **kwargs):
    """Print if in debug mode"""
    if VERBOSE:
        print(*args, **kwargs)
