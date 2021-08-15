
import numpy as np
import pandas as pd
import os

import explore_data, feature_engineering, train_with_ML_algos, train_with_nn
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

def read_data(path):
    '''
    Read csv file and return pandas dataframe
    Input:
        path: csv file path
    Output:
        data: pandas dataframe
    '''
    try:
        print('UPLOADING', path, ' ...')
        data_frame = pd.read_csv(path)
        return data_frame
    except FileNotFoundError:
        print('Eror in finding file')

def perform_data_exploration(data_frame):
    '''
    Perfrom data exploration on data_frame and save figures to images folder
    input:
        data_frame: pandas dataframe

    output:
        None
    '''
    explore_data.display_boxplot('Pclass', 'Embarked', data_frame, 'pclass_embarked_boxplot.png', 'h')
    explore_data.display_boxplot('Fare', 'Pclass', data_frame, 'fare_pclass_boxplot.png', 'h')

    columns_lst = ["SibSp","Parch","Survived","Pclass", "Age","Fare"] 
    explore_data.display_heatmap(columns_lst, data_frame, 'heatmap.png')

    explore_data.display_factorplot('SibSp', 'Survived', data_frame, 'sibsap_survived_factorplot.png') 
    explore_data.display_factorplot('Parch', 'Survived', data_frame, 'sibsap_survived_factorplot.png') 
    explore_data.display_factorplot('Pclass', 'Survived', data_frame, 'pclass_survived_factorplot.png')

    explore_data.display_displot('Age', 'Survived', data_frame, 'age_survived_displot.png')
    explore_data.display_displot('Fare', 'Survived', data_frame, 'fare_survived_displot.png')

    explore_data.display_factorplot('Sex', 'Age', data_frame, 'sex_age_factorplot.png') 
    explore_data.display_factorplot('SibSp', 'Age', data_frame, 'sibsap_age_factorplot.png') 
    explore_data.display_factorplot('Parch', 'Age', data_frame, 'parch_age_factorplot.png') 

    # explore_data.display_countplot('Title', data_frame, 'title_countplot.png') 
    # explore_data.display_countplot('Title', data_frame, 'title_countplot.png') 

def perform_feature_engineering(data_frame):

    print('DATA PREPROCESSING ...')
    data_frame = feature_engineering.preprocess_embarked(data_frame)
    data_frame = feature_engineering.preprocess_fare(data_frame)
    data_frame = feature_engineering.preprocess_age(data_frame)
    data_frame = feature_engineering.preprocess_name(data_frame)
    data_frame = feature_engineering.preprocess_sibsp_parch(data_frame)
    data_frame = feature_engineering.preprocess_tickets(data_frame)
    data_frame = feature_engineering.preprocess_pclass(data_frame)
    data_frame = feature_engineering.preprocess_sex(data_frame)
    print('Data Preprocessing was successfully done-------')

    data_frame.drop(labels = ["PassengerId", "Cabin", "Name"], axis = 1, inplace = True)
    train_test_valid_data = feature_engineering.prepare_train_test_data(data_frame, df_train_len)
    return train_test_valid_data

def evaluate_model(model, X_valid, y_valid):
    '''
    Evaluate the performance using validation data and display the results
    Input:
        model: classification model
        X_valid: features for validation data
        y_valid: targets for validation data
    '''
    y_pred = model.predict_classes(X_valid)
    acc_score = accuracy_score(y_valid, y_pred)
    print("Accuracy Score:", acc_score)
    print("Classification Report:")
    print(classification_report(y_valid, y_pred))

def evaluate_and_save_submission(model, test_data):
    '''
    Evaluate the performance and save the submission file
    Input:
        model: classification model
        test_data: submission test data
    Output:
        y_pred: prediction result
    '''
    y_pred = pd.Series(list(model.predict_classes(test_data)), name = "Survived").astype(int)
    results = pd.concat([test_passengerId, y_pred],axis = 1)
    
    results.to_csv("submission.csv", index = False)

if __name__ == '__main__':
    df_train = read_data('train.csv')
    df_test = read_data('test.csv')
    print('Uploaded successfully---------')

    test_passengerId = df_test["PassengerId"]

    df_train_len = len(df_train)
    df_train = pd.concat([df_train,df_test],axis = 0).reset_index(drop = True)

    # perform_data_exploration(df_train)
    [test, [X_train, X_valid, y_train, y_valid]] = perform_feature_engineering(df_train)
    print('Test data shape', test.shape)
    print('X_train.shape{} y_train.shape{} X_valid.shape{} y_valid.shape{}'.format(
        X_train.shape, y_train.shape, X_valid.shape, y_valid.shape
    ))

    model = build_model()
    model.summary()