import torch
from torch.utils.data import DataLoader

from data import LibriSpeechDataset, LIBRISPEECH_SAMPLING_RATE


def prep():
    ##############
    n_seconds = 3
    downsampling = 1
    batchsize = 8
    # model_type = 'max_pooling'
    # model_n_layers = 7
    # model_n_filters = 64
    # model_dilation_depth = 7  # Only relevant for model_type == 'dilated'
    # model_dilation_stacks = 1  # Only relevant for model_type == 'dilated'
    training_set = ['train-clean-100', 'train-clean-360']
    validation_set = 'dev-clean'
    learning_rate = 0.005
    momentum = 0.9
    n_epochs = 10
    reduce_lr_patience = 32
    evaluate_every_n_batches = 500

    ###################
    # Create datasets #
    ###################
    trainset = LibriSpeechDataset(training_set, int(LIBRISPEECH_SAMPLING_RATE * n_seconds))
    testset = LibriSpeechDataset(validation_set, int(LIBRISPEECH_SAMPLING_RATE * n_seconds), stochastic=False)
    trainloader = DataLoader(trainset, batch_size=batchsize, num_workers=4, shuffle=True, drop_last=True)
    testloader = DataLoader(testset, batch_size=batchsize, num_workers=4, drop_last=True)

    return trainloader, testloader


def run(trainloader, testloader):
    torch.multiprocessing.freeze_support()

    for step, (x, y) in enumerate(trainloader):  # gives batch data
        print(x, y)


if __name__ == '__main__':
    x,y=prep()
    run(x,y)
