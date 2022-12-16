import numpy as np
import torch
from math import floor, prod
from keras.datasets import cifar100
from keras.utils import to_categorical


def get_CIFAR_100():
    """
    Dataset of 50.000 32x32 color training images, labeled over 100 categories, and 10,000 test images.

    :return: tuple of X_train, y_train, X_test, y_test
    """
    (X_train, y_train), (X_test, y_test) = cifar100.load_data(label_mode='fine')
    X_train = np.moveaxis(X_train, 3, 1)    # channels last to channels first
    X_test = np.moveaxis(X_test, 3, 1)      # channels last to channels first
    return X_train, y_train, X_test, y_test


def disjoint_datasets(X, y):
    """
    Separate bigger dataset to 10 smaller datasets.

    :param X: model input data
    :param y: model output data / label
    :return: 10 disjoint datasets
    """
    sets = [([], []) for _ in range(10)]
    for image, label in zip(X, y):
        index = int(floor(label[0] / 10))
        sets[index][0].append(image)
        sets[index][1].append(to_categorical(label[0] % 10, 10))
    return sets


def make_disjoint_datasets(dataset_fun=get_CIFAR_100):
    """
    Make 10 disjoint datasets of the same size from CIFAR-100.

    :param dataset_fun: function that returns specific dataset (default is CIFAR-100 dataset)
    :return: list of 10 disjoint datasets with corresponding train and test set
             [(X_train, y_train, X_test, y_test), (X_train, y_train, X_test, y_test), ...]
    """
    X_train, y_train, X_test, y_test = dataset_fun()
    train_sets = disjoint_datasets(X_train, y_train)
    test_sets = disjoint_datasets(X_test, y_test)
    return list(map(lambda x: (*x[0], *x[1]), zip(train_sets, test_sets)))


def get_dataset(nn_cnn, input_size):
    """
    Prepare dataset for input to NN or CNN.

    :param nn_cnn: string: 'nn' or 'cnn'
    :param input_size: image input size in pixels
    :return: (X_train, y_train, X_test, y_test) of 10 disjoint sets of CIFAR-100
    """
    disjoint_sets = make_disjoint_datasets()
    for i, dis_set in enumerate(disjoint_sets):
        X_train, y_train, X_test, y_test = dis_set

        # normalize input images to have values between 0 and 1
        X_train = np.array(X_train).astype(dtype=np.float64)
        X_test = np.array(X_test).astype(dtype=np.float64)
        X_train /= 255
        X_test /= 255

        if nn_cnn == 'nn':
            # reshape to the right dimensions for NN
            X_train = X_train.reshape(X_train.shape[0], prod(input_size))
            X_test = X_test.reshape(X_test.shape[0], prod(input_size))

        y_train = np.array(y_train)
        y_test = np.array(y_test)

        disjoint_sets[i] = (torch.tensor(X_train).float(), torch.tensor(y_train).float(),
                            torch.tensor(X_test).float(), torch.tensor(y_test).float())

    return disjoint_sets


if __name__ == '__main__':
    d = get_dataset('nn', (3, 32, 32))
