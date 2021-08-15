from keras import callbacks
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input, BatchNormalization
from keras import optimizers

def build_model():
    '''
    Build neural network for classification
    Output:
    model: neural network model
    '''

    model = Sequential([
        Input(shape=(X_train.shape[1])),
        Dense(32, activation='relu'),
        Dropout(0.3),
        BatchNormalization(),
        
        Dense(16, activation='relu'),
        Dropout(0.3),
        BatchNormalization(),
        
        Dense(5, activation='relu'),
        BatchNormalization(),
        
        Dense(1, activation="sigmoid")
    ])
    return model

def train(X_train, y_train, X_valid, y_valid, model, batch_size, epochs):
    '''
    Compile and fit the model using Adam optimizer
    INPUT:
    model: neural network model
    '''
    adam = optimizers.Adam(lr = 0.0001)
    model.compile(loss="binary_crossentropy", optimizer=adam, metrics=["accuracy"])
    early_stop = callbacks.EarlyStopping(patience=20)

    history = model.fit(
        X_train, y_train,
        batch_size = batch_size,
        epochs=epochs, validation_data=(X_valid, y_valid), 
        callbacks=[early_stop],
        verbose=1
    )

    df_plot = pd.DataFrame(history.history)
    df_plot.plot(figsize=(10,5))
    df_plot.savefig('./images/classifier/classifiers_mean_acc.png')