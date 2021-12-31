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

    ###################
    # Create datasets #
    ###################
    trainset = LibriSpeechDataset(training_set, int(LIBRISPEECH_SAMPLING_RATE * n_seconds))
    # testset = LibriSpeechDataset(validation_set, int(LIBRISPEECH_SAMPLING_RATE * n_seconds), stochastic=False)
    trainloader = DataLoader(trainset, batch_size=batchsize, num_workers=4, shuffle=True, drop_last=True)
    # trainloader = DataLoader(trainset, num_workers=4, shuffle=True, drop_last=True)
    # testloader = DataLoader(testset, batch_size=batchsize, num_workers=4, drop_last=True)
    # testloader = DataLoader(testset, num_workers=4, drop_last=True)

    return trainloader  # , testloader


def get_features(trainloader):
    torch.multiprocessing.freeze_support()
    features = []  # list to save features
    labels = []  # list to save labels

    for step, (data, label) in enumerate(trainloader):  # gives batch data
        features.append(data)
        labels.append(label)
        if step == 0:
            break
    output = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    print(labels)

    return np.array(output), labels


# print(x, y)
# print (x[0].shape)
#
# break


if __name__ == '__main__':
    data_loader = prep()
    X, y = get_features(data_loader)
    # cast to np array
    X = np.array((X - np.min(X)) / (np.max(X) - np.min(X)))
    X = X / np.std(X)
    y = np.array(y)
    print(X.shape, y.shape)

    # Split twice to get the validation set
    print(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=123, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=123)
    # Print the shapes
    print(X_train.shape, X_test.shape, X_val.shape, len(y_train), len(y_test), len(y_val))
    # input_shape = 128
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
    input_shape = (128,94)

    model = keras.Sequential()
    model.add(LSTM(256, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(2, activation='softmax'))

    model.summary()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=50, batch_size=72,
                        validation_data=(X_val, y_val), shuffle=False)
    model.save("my_model")
    # reconstructed model = keras.models.load_model("my_model");
    #check
    #np.testing.assert_allclose( model.predict(test_input),reconstructed_model.predict(test_input))
