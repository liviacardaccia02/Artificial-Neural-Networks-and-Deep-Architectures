import numpy as np
import os
from part2_timeseries_livia.build_dataset import make_embedding, split_data, add_noise
from part2_timeseries_livia.generate_mackey_glass import mackey_glass

os.environ.setdefault("KERAS_BACKEND", "torch")

import keras
import torch
from keras.regularizers import l2

def build_and_train_model(X_train, Y_train, X_val, Y_val, nh1, nh2, learning_rate=0.01, l2_reg=0.001, patience=3000, epochs=6000):
    model = keras.Sequential([keras.layers.Input(shape=(5,)),
                              keras.layers.Dense(nh1, activation='tanh', kernel_regularizer=l2(0.001)),    
                              keras.layers.Dense(nh2, activation='tanh', kernel_regularizer=l2(0.001)),
                              keras.layers.Dense(1, activation='linear')])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=[keras.metrics.MeanSquaredError(name="mse")] 
    )

    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_mse',
        patience=patience,
        restore_best_weights=True,
        mode='min'
    )

    history = model.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    epochs=epochs,          
    batch_size=len(X_train), # Full batch descent
    callbacks=[early_stop],
    verbose=0              
    )


nh1 = [3, 4, 5]
nh2 = [2, 4, 6]

X = mackey_glass(n_points=1600)

X_noise = add_noise(X, sigma=0.05, seed=42)

X_embedded, Y_embedded = make_embedding(X_noise, t_start=301, t_end=1500)

X_train, Y_train, X_val, Y_val, X_test, Y_test = split_data(X_embedded, Y_embedded, n_train=700, n_val=300, n_test=200)

for n1 in nh1:
    for n2 in nh2:
        print(f"Training model with {n1} neurons in the first hidden layer and {n2} neurons in the second hidden layer.")
        build_and_train_model(X_train, Y_train, X_val, Y_val, n1, n2)