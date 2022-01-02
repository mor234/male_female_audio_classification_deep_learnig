import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader

from data import LibriSpeechDataset, LIBRISPEECH_SAMPLING_RATE


def prep():
    ##############
    n_seconds = 3
    batchsize = 20000

    training_set = ['train-clean-100', 'train-clean-360']  # ,
    test_set = ['dev-clean', 'test-clean']

    ###################
    # Create datasets #
    ###################
    trainset = LibriSpeechDataset(training_set, int(LIBRISPEECH_SAMPLING_RATE * n_seconds))
    testset = LibriSpeechDataset(test_set, int(LIBRISPEECH_SAMPLING_RATE * n_seconds), stochastic=False)
    trainloader = DataLoader(trainset, batch_size=batchsize, num_workers=4, shuffle=True, drop_last=True)
    testloader = DataLoader(testset, batch_size=batchsize, num_workers=4, drop_last=True)

    return trainloader  # ,trainloader


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


if __name__ == '__main__':
    data_loader = prep()
    X, y = get_features(data_loader)
    print(X.shape, y.shape)

    # Split twice to get the validation set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=123, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=123)
    print("before:", X, y)

    # normelize
    norm = MinMaxScaler().fit(X_train)
    X_train = norm.transform(X_train)
    X_val = norm.transform(X_val)
    X_test=norm.transform(X_test)

    print("after:", X_train)  # ,X_val)

    # Print the shapes
    # print(X_train.shape, X_test.shape, X_val.shape, len(y_train), len(y_test), len(y_val))

    # save to csv file
    np.savetxt("x_train1.csv", X_train)
    np.savetxt("y_train1.csv", y_train)
    np.savetxt("x_test.csv", X_test)
    np.savetxt("y_test.csv", y_test)
    np.savetxt("x_val1.csv", X_val)
    np.savetxt("y_val1.csv", y_val)

    loaded_arr0 = np.loadtxt("x_test.csv")
    loaded_arr1 = np.loadtxt("y_test.csv")

    if (y_test == loaded_arr1).all():
        print("1: Yes, both the arrays are same")
    if (loaded_arr0 == X_test).all():
        print("2: Yes, both the arrays are same")
