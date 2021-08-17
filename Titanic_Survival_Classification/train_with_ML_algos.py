import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

def define_hyperparameters(random_state = 42):
    '''
    Define hyperparamters for classifiers
    dt_param_grid: hyperparameters for decision tree
    svc_param_grid: hyperparameters for support vector machine
    rf_param_grid: hyperparameters for random forest
    logreg_param_grid: hyperparameters for logistics regression
    knn_param_grid: hyperparameters for k-nearest neighbor classifier

    INPUT:
        random_state: parameter for random_state of classifiers
    OUTPUT:
        classifier: classifiers for the problems
        classifier_param: hyperparameters for classifier

    '''
    classifier = [DecisionTreeClassifier(random_state = random_state),
                SVC(random_state = random_state),
                RandomForestClassifier(random_state = random_state),
                LogisticRegression(random_state = random_state),
                KNeighborsClassifier()]

    dt_param_grid = {"min_samples_split" : range(10,500,20),
                    "max_depth": range(1,20,2)}

    svc_param_grid = {"kernel" : ["rbf"],
                    "gamma": [0.001, 0.01, 0.1, 1],
                    "C": [1,10,50,100,200,300,1000]}

    rf_param_grid = {"max_features": [1,3,5,10],
                    "min_samples_split":[2,3,5,10],
                    "min_samples_leaf":[1,3,5,10],
                    "bootstrap":[False],
                    "n_estimators":[100,300,10],
                    "criterion":["gini"]}

    logreg_param_grid = {"C":np.logspace(-3,3,7),
                        "penalty": ["l1","l2"]}

    knn_param_grid = {"n_neighbors": np.linspace(1,19,10, dtype = int).tolist(),
                    "weights": ["uniform","distance"],
                    "metric":["euclidean","manhattan"]}
    classifier_param = [dt_param_grid,
                    svc_param_grid,
                    rf_param_grid,
                    logreg_param_grid,
                    knn_param_grid]

    return classifier, classifier_param


def find_best_classifiers(X_train, y_train, classifier, classifier_param):
    '''
    Train the classifers and find the best parameter
    INPUT:
        classifier: list of the classifiers
        classifier_param: hyperparameters of the clasifier

    OUTPUT:
        cv_result: list of the score of classifiers
        best_estimators: best estimator of classifiers
    '''
    cv_result = []
    best_estimators = []
    for i in range(len(classifier)):
        clf = GridSearchCV(classifier[i], param_grid=classifier_param[i], cv = StratifiedKFold(n_splits = 10), scoring = "accuracy", n_jobs = -1,verbose = 1)
        clf.fit(X_train,y_train)
        cv_result.append(clf.best_score_)
        best_estimators.append(clf.best_estimator_)
        print(cv_result[i])
    return cv_result, best_estimators

def plot_classifiers(cv_result):
    '''
    Plot the result of classifiers
    INPUT:
        cv_result: list of the score of classifiers
    OUTPUT:
        NONE
    '''
    cv_df = pd.DataFrame({"Cross Validation Means":cv_result, "ML Models":["DecisionTreeClassifier", "SVM","RandomForestClassifier",
             "LogisticRegression",
             "KNeighborsClassifier"]})

    plt.figure(figsize=(20, 10))
    sns.barplot("Cross Validation Means", "ML Models", data = cv_df)
    plt.savefig('./images/classifier/classifiers_mean_acc.png')

def train_ensemble_model(best_estimators, X_train, y_train):
    '''
    Train ensemble model using voting classifier
    INPUT:
        cv_result: list of the score of classifiers
        classifier: list of the classifiers
        classifier_param: hyperparameters of the clasifier
    '''
    ensemble_model = VotingClassifier(estimators = [("dt",best_estimators[0]),
                                        ("svm",best_estimators[1]),
                                        ("rfc",best_estimators[2])],
                                        voting = "hard", n_jobs = -1)
    ensemble_model = ensemble_model.fit(X_train, y_train)

    return ensemble_model