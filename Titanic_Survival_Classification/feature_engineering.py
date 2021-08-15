import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def preprocess_embarked(data_frame):
    '''
    Fill null value of Embarked with 'C' after exploration and return dataframe
    input:
        data: pandas series
    output:
        data_frame: pandas dataframe
    '''
    data_frame["Embarked"] = data_frame["Embarked"].fillna("C")
    data_frame = pd.get_dummies(data_frame, columns=["Embarked"])
    return data_frame

def preprocess_fare(data_frame):
    '''
    Fill null value of Fare with mean value after exploration and return dataframe
    input:
        data_frame: pandas series
    output:
        data_frame: pandas dataframe
    '''
    data_frame["Fare"] = data_frame["Fare"].fillna(np.mean(data_frame[data_frame["Pclass"] == 3]["Fare"]))
    return data_frame

def preprocess_age(data_frame):
    '''
    Find values of age to fill for null value after exploration and return dataframe
    input:
        data_frame: pandas series
    output:
        data_frame: pandas dataframe
    '''
    index_nan = list(data_frame[data_frame.Age.isnull()].index)

    for i in index_nan:
        age_pred = data_frame["Age"][((data_frame["SibSp"] == data_frame.iloc[i]["SibSp"]) & (data_frame["Parch"] == data_frame.iloc[i]["Parch"]) & 
                                (data_frame["Pclass"] == data_frame.iloc[i]["Pclass"]))].median()
        age_med = data_frame["Age"].median()
        
        if not np.isnan(age_pred):
            data_frame["Age"].iloc[i] = age_pred
        else:
            data_frame["Age"].iloc[i] = age_med 
    return data_frame

def preprocess_name(data_frame):
    '''
    Process Name value and return dataframe
    input:
        data_frame: pandas series
    output:
        data_frame: pandas dataframe
    '''   
    names = data_frame["Name"]
    data_frame["Title"] = [name.split(".")[0].split(",")[-1].strip() for name in names]
    data_frame["Title"] = data_frame["Title"].replace(["Lady","the Countess","Capt","Col","Don","Dr","Major","Rev","Sir","Jonkheer","Dona"],"Other")
    data_frame["Title"] = [0 if i == "Master" else 1 if i == "Miss" or i == "Ms" or i == "Mlle" or i == "Mrs" else 2 if i == "Mr" else 3 for i in data_frame["Title"]]

    data_frame = pd.get_dummies(data_frame,columns=["Title"])
    return data_frame

def preprocess_sibsp_parch(data_frame):
    '''
    Process SibSp and Parch values, create a new column called Family size and return dataframe
    input:
        data_frame: pandas series
    output:
        data_frame: pandas dataframe
    ''' 
    data_frame["Family_size"] = data_frame["SibSp"] + data_frame["Parch"] + 1
    data_frame["Family_size"] = [1 if i < 5 else 0 for i in data_frame["Family_size"]]
    data_frame = pd.get_dummies(data_frame, columns= ["Family_size"])
    return data_frame

def preprocess_tickets(data_frame):
    '''
    Process Tickets value and return dataframe
    input:
        data_frame: pandas series
    output:
        data_frame: pandas dataframe
    ''' 
    tickets = []
    for i in list(data_frame.Ticket):
        if not i.isdigit():
            tickets.append(i.replace(".","").replace("/","").strip().split(" ")[0])
        else:
            tickets.append("x")
    data_frame["Ticket"] = tickets
    data_frame = pd.get_dummies(data_frame, columns= ["Ticket"], prefix = "T")
    return data_frame

def preprocess_pclass(data_frame):
    '''
    Process Pclass value and return dataframe
    input:
        data_frame: pandas series
    output:
        data_frame: pandas dataframe
    ''' 
    data_frame["Pclass"] = data_frame["Pclass"].astype("category")
    data_frame = pd.get_dummies(data_frame, columns= ["Pclass"])
    return data_frame

def preprocess_sex(data_frame):
    '''
    Process Sex value and return dataframe
    input:
        data_frame: pandas series
    output:
        data_frame: pandas dataframe
    ''' 
    data_frame["Sex"] = data_frame["Sex"].astype("category")
    data_frame = pd.get_dummies(data_frame, columns=["Sex"])
    return data_frame

def prepare_train_test_data(data_frame, train_data_len):
    '''
    Prepare data as train and test from pandas dataframe, perform train_test_split and return necessary data
    input:
        data_frame: pandas series
        train_data_len: actual size of training data
    output:
        [test, [X_train, X_valid, y_train, y_valid]]: an array of actual test data, train/valid features, train/valid labels
    '''  
    test = data_frame[train_data_len:]
    test.drop(labels = ["Survived"],axis = 1, inplace = True)

    train = data_frame[:train_data_len]

    X_train = train.drop(labels = "Survived", axis = 1)
    y_train = train["Survived"]

    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size = 0.3, random_state = 42)
    return test, [X_train, X_valid, y_train, y_valid]