# AI for beginners

![Screen Shot 2021-08-17 at 23 29 54](https://user-images.githubusercontent.com/20230956/129769155-903da1ce-8f88-407d-9761-2043e43550a6.png)


## Table of Contents
* [General Info](#general-info)
* [Repositories](#repositories)
* [Technologies](#technologies)

## General info
This repo is for those who would like to start learning machine learning algorithms and the machine pipeline.

## Repositories
* Titanic Survival Classification

[Project Background] 
This is for survival prediction on Titanic Data. The model is built to predict which passenger survived on Titanic shipwreck.


[Dataset]

![df](https://user-images.githubusercontent.com/20230956/129834961-482e7232-5f74-4f37-a583-b96435b0e5f1.png)

[Model]
```
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense (Dense)                (None, 32)                1792      
_________________________________________________________________
dropout (Dropout)            (None, 32)                0         
_________________________________________________________________
batch_normalization (BatchNo (None, 32)                128       
_________________________________________________________________
dense_1 (Dense)              (None, 16)                528       
_________________________________________________________________
dropout_1 (Dropout)          (None, 16)                0         
_________________________________________________________________
batch_normalization_1 (Batch (None, 16)                64        
_________________________________________________________________
dense_2 (Dense)              (None, 5)                 85        
_________________________________________________________________
batch_normalization_2 (Batch (None, 5)                 20        
_________________________________________________________________
dense_3 (Dense)              (None, 1)                 6         
=================================================================
Total params: 2,623
Trainable params: 2,517
Non-trainable params: 106
```

[Evaluation]

Classification Report:
```
                precision   recall    f1-score   support

    0.0         0.78      0.93        0.85       157
    1.0         0.86      0.63        0.73       111

accuracy                              0.81       268
macro avg       0.82        0.78      0.79       268
weighted avg    0.82        0.81      0.80       268

```
## Technologies
* Pandas, numpy, os,
* matplotlib, seaborn, 
* tensorflow, scikit-learn, keras
