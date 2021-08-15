import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use("seaborn-darkgrid")

import warnings 
warnings.filterwarnings("ignore")

def display_boxplot(x, y, data, file_name, orient = "v"):
    '''
    Display boxplot for the given data
    input:        
        x: x axis value
        y: y axis value
        data: pandas series
        
        orientat: orientation of plot
        file_name: string
    output:
        NONE
    '''
    plt.figure(figsize=(20, 10))
    sns.boxplot(x=x, y=y, data=data, orient=orient, palette="Set2")
    plt.savefig('./images/exploratory/' + file_name)

def display_heatmap(columns_lst, data, file_name):
    '''
    Display heatmap for the given data
    input:
        columns_lst: list of the columns
        data: pandas series
        file_name: string
    output:
        NONE
    '''
    plt.figure(figsize=(20, 10))
    sns.heatmap(data[columns_lst].corr(),annot = True,fmt = ".2f")
    plt.savefig('./images/exploratory/' + file_name)

def display_factorplot(x, y, data, file_name):
    '''
    Display factorplot for the given data
    input:
        x: x axis value
        y: y axis value 
        data: pandas series
        file_name: string
    output:
        NONE
    '''
    plt.figure(figsize=(20, 10))
    sns.factorplot(x = x, y = y,kind = "bar",size = 7,data = data)
    plt.savefig('./images/exploratory/' + file_name)

def display_displot(x, y, data, file_name):
    '''
    Display displot for the given data
    input:
        x: x axis value
        y: y axis value 
        data: pandas series
        file_name: string
    output:
        NONE
    '''
    plt.figure(figsize=(20, 10))
    fare_survival = sns.FacetGrid(data, col = y,size = 7)
    fare_survival.map(sns.distplot, x, bins = 10)
    plt.savefig('./images/exploratory/' + file_name)

def display_countplot(x, data, file_name):
    '''
    Display countplot for the given data
    input:
        x: x axis value
        data: pandas series
        file_name: string
    output:
        NONE
    '''
    plt.figure(figsize=(20, 10))
    sns.countplot(x=x, data = data)
    plt.savefig('./images/exploratory/' + file_name)




