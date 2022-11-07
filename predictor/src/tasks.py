from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB, ComplementNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from data import *
from utils import *

# RQ.1: Jaki wpływ na dokładność predykcji ma wykorzystanie nowego zestawu cech ?
def task1():
    skf = KFold(n_splits=FOLDS, random_state=SEED, shuffle=True)
    x_test, y_test, _ = prepare_data(DIR_LTC3_2017)

    classifiers = [
        ("Random Forest", RandomForestClassifier(random_state=SEED, n_jobs=-1)),
        ("Nearest Neighbour", KNeighborsClassifier(n_jobs=-1 )),
        ("Decision Tree", DecisionTreeClassifier(random_state=SEED)),
        ("Gaussian Naive Bayes", GaussianNB())
    ]

    for name, clf in classifiers:
        test_classifier(name, x_test, y_test, clf, skf)

# RQ.2: Czy wykorzystanie nowych klasyfikatorów wpłynie pozytywnie na otrzymywane rezultaty ?
def task2():
    skf = KFold(n_splits=FOLDS, random_state=SEED, shuffle=True)
    x_test, y_test, _ = prepare_data(DIR_LTC3_2017)

    classifiers = [
        ("Complement Naive Bayes", ComplementNB()),
        ("Neural Network", MLPClassifier(random_state=SEED)),
        ("Gradient Boosting", GradientBoostingClassifier(random_state=SEED))
    ]

    for name, clf in classifiers:
        test_classifier(name, x_test, y_test, clf, skf)


# RQ.3: Jak wpływa przyjęty próg czasowy określający twórcę jako „długookresowego” na wyniki predykcji ?
def task3():
    skf = KFold(n_splits=FOLDS, random_state=SEED, shuffle=True)

    datasets = [
        DIR_LTC1_2017,
        DIR_LTC2_2017,
        DIR_LTC3_2017,
        DIR_LTC4_2017,
        DIR_LTC1_2018,
        DIR_LTC2_2018,
        DIR_LTC3_2018,
        DIR_LTC4_2018,
    ]

    classifiers = [
        ("Random Forest", RandomForestClassifier(random_state=SEED, n_jobs=-1)),
        ("Gradient Boosting", GradientBoostingClassifier(random_state=SEED))
    ]

    for dataset_name in datasets:
        x_test, y_test, _ = prepare_data(dataset_name)
        for clf_name, clf in classifiers:
            f_name = dataset_name.split('/')[-1].split('.')[0]
            name = f_name + ' ' + clf_name
            test_classifier(name, x_test, y_test, clf, skf)

# RQ.4: Czy strojenie hiperparametrów modelu poprawi otrzymywane rezultaty ?
def task4():
    x_train, x_test, y_train, y_test, _ = prepare_and_split_data(DIR_LTC3_2017)

    # tuning_rf(x_train, y_train)
    # tuning_gb(x_train, y_train)
    # return
    classifiers = [
        ("Random Forest", RandomForestClassifier(random_state=SEED, n_jobs=-1, n_estimators=600,
                                                 min_samples_split=2, min_samples_leaf=3, max_leaf_nodes=None,
                                                 max_features='sqrt', max_depth=10, criterion='gini',
                                                 class_weight='balanced_subsample', bootstrap=True)),
        ("Gradient Boosting", GradientBoostingClassifier(random_state=SEED, n_estimators=1000, min_samples_split=2,
                                                         min_samples_leaf=2, max_leaf_nodes=None, max_features='sqrt',
                                                         max_depth=4, loss='exponential', learning_rate=0.1)),
    ]

    skf = KFold(n_splits=FOLDS, random_state=SEED, shuffle=True)
    for name, clf in classifiers:
        test_classifier(name, x_test, y_test, clf, skf)

# RQ.5: Które z cech są najistotniejsze dla opracowanych modeli ?
def task5():
    x_train, _, y_train, _, _ = prepare_and_split_data(DIR_LTC3_2017)
    classifiers = [
        ("Random Forest", RandomForestClassifier(random_state=SEED, n_jobs=-1, n_estimators=600,
                                                 min_samples_split=2, min_samples_leaf=3, max_leaf_nodes=None,
                                                 max_features='sqrt', max_depth=10, criterion='gini',
                                                 class_weight='balanced_subsample', bootstrap=True)),
        ("Gradient Boosting", GradientBoostingClassifier(random_state=SEED, n_estimators=1000, min_samples_split=2,
                                                         min_samples_leaf=2, max_leaf_nodes=None, max_features='sqrt',
                                                         max_depth=4, loss='exponential', learning_rate=0.1)),
    ]

    for name, clf in classifiers:
        feature_importance_rank(name, x_train, y_train, clf)

def tuning_rf(x, y):
    y = y.values.ravel()
    skf = KFold(n_splits=5, random_state=SEED, shuffle=True)
    clf = RandomForestClassifier(random_state=SEED, n_jobs=-1)

    # best
    # score: 0.3919
    # best
    # param: {'n_estimators': 600, 'min_samples_split': 2, 'min_samples_leaf': 3, 'max_leaf_nodes': None,
    #         'max_features': 'sqrt', 'max_depth': 10, 'criterion': 'gini', 'class_weight': 'balanced_subsample',
    #         'bootstrap': True}

    # rf_grid = {
    #     'bootstrap': [False, True],
    #     'max_depth': [10, 18, 28, 50, None],
    #     'max_features': ['sqrt', 'log2', None],
    #     'min_samples_leaf': [1, 2, 3, 4],
    #     'min_samples_split': [2, 5, 10],
    #     'n_estimators': [50, 150, 300, 600, 1000],
    #     'class_weight': ['balanced', 'balanced_subsample'],
    #     'criterion': ['entropy', 'gini'],
    #     'max_leaf_nodes': [30, 60, 120, None],
    # }
    # # 5000 of 28800 combinations so grid can be small in next step
    # optimize_for_random("Random Forest", x, y, clf, skf, rf_grid, mcc_scorer(), 5000, SEED)

    # python_1 | best
    # score: 0.3919
    # python_1 | best
    # param: {'bootstrap': True, 'class_weight': 'balanced_subsample', 'criterion': 'gini', 'max_depth': 10,
    #         'max_features': 'sqrt', 'max_leaf_nodes': None, 'min_samples_lea
    #                                                         f': 3, 'min_samples_split': 2, 'n_estimators': 600}

    rf_grid = {
        'bootstrap': [True, False],
        'max_depth': [9, 10, 11],
        'max_features': ['sqrt'],
        'min_samples_leaf': [3],
        'min_samples_split': [2],
        'n_estimators': [550, 600, 650],
        'class_weight': ['balanced_subsample'],
        'criterion': ['gini', 'entropy'],
        'max_leaf_nodes': [None],
    }
    optimize_for_grid("Random Forest", x, y, clf, skf, rf_grid, mcc_scorer())


def tuning_gb(x, y):
    y = y.values.ravel()
    skf = KFold(n_splits=5, random_state=SEED, shuffle=True)
    clf = GradientBoostingClassifier(random_state=SEED)

    # best
    # score: 0.3579
    # best
    # param: {'n_estimators': 1000, 'min_samples_split': 2, 'min_samples_leaf': 2, 'max_leaf_nodes': None,
    #         'max_features': 'sqrt', 'max_depth': 4, 'loss': 'exponential', 'learning_rate': 0.1}

    # gb_grid = {
    #     'loss': ['deviance', 'exponential'],
    #     'learning_rate': [0.05, 0.1, 0.2],
    #     'n_estimators': [50, 150, 300, 600, 1000],
    #     'max_depth': [2, 3, 4, None],
    #     'min_samples_leaf': [1, 2, 3],
    #     'min_samples_split': [2, 5],
    #     'max_features': ['sqrt', 'log2', None],
    #     'max_leaf_nodes': [30, 60, 120, None],
    # }
    # # 3000 of 8640 combinations so grid can be small in next step
    # optimize_for_random("Gradient Boosting", x, y, clf, skf, gb_grid, mcc_scorer(), 3000, SEED)

    # best
    # score: 0.35792
    # best
    # param: {'learning_rate': 0.1, 'loss': 'exponential', 'max_depth': 4, 'max_features': 'sqrt', 'max_leaf_nodes': None,
    #         'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 1000}

    gb_grid = {
        'loss': ['exponential'],
        'max_depth': [4, 5, 6, None],
        'max_features': ['sqrt'],
        'min_samples_leaf': [2],
        'min_samples_split': [2, 3, 4],
        'n_estimators': [900, 1000, 1200],
        'learning_rate': [0.1],
        'max_leaf_nodes': [None],
    }
    optimize_for_grid("Gradient Boosting", x, y, clf, skf, gb_grid, mcc_scorer())
