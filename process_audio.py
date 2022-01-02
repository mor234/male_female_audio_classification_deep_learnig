import numpy as np
import torch
from keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from tensorflow import keras
from torch.utils.data import DataLoader

from data import LibriSpeechDataset, LIBRISPEECH_SAMPLING_RATE


def prep():
    ##############
    n_seconds = 3
    downsampling = 1
    batchsize = 20000
    # model_type = 'max_pooling'
    # model_n_layers = 7
    # model_n_filters = 64
    # model_dilation_depth = 7  # Only relevant for model_type == 'dilated'
    # model_dilation_stacks = 1  # Only relevant for model_type == 'dilated'
    training_set = ['train-clean-100', 'train-clean-360']#, 'dev-clean']
    # validation_set = 'dev-clean'
    learning_rate = 0.005
    momentum = 0.9
    n_epochs = 10
    reduce_lr_patience = 32
    evaluate_every_n_batches = 500




if __name__ == '__main__':
    X_train = np.loadtxt("x_train.csv")
    X_val = np.loadtxt("x_val.csv")
    y_train = np.loadtxt("y_train.csv")
    y_val = np.loadtxt("y_val.csv")
    # Print the shapes
    print(X_train.shape, X_val.shape, len(y_train),len(y_val))
    input_shape = (128,1)
    # model = keras.Sequential()
    #   model.add(LSTM(128, input_shape=input_shape))
    #   model.add(Dropout(0.2))
    #   model.add(Dense(128, activation='relu'))
    #   model.add(Dense(64, activation='relu'))
    #   model.add(Dropout(0.4))
    #   model.add(Dense(48, activation='relu'))
    #   model.add(Dropout(0.4))
    #   model.add(Dense(2, activation='softmax'))
    #
    # input_shape = (128,94)
    #
    # model = keras.Sequential()
    # model.add(LSTM(256, input_shape=input_shape))
    # model.add(Dropout(0.2))
    # model.add(Dense(128, activation='relu'))
    # model.add(Dropout(0.4))
    # model.add(Dense(2, activation='softmax'))
    #
    # model.summary()
    opt = keras.optimizers.Adam(learning_rate=0.001)

    model = keras.models.load_model("san2")
    # model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=50, batch_size=72,
                        validation_data=(X_val, y_val), shuffle=False)
    model.save("san3")
    # reconstructed model = keras.models.load_model("my_model");
    #check
    #np.testing.assert_allclose( model.predict(test_input),reconstructed_model.predict(test_input))
