# This script serves as helpers to simplify some of the code needed to do various functions for the jupyter notebook
# The thought is that this should streamline the workshop a bit better, and avoiding tricky python specific code
# for participants who might not be too familiar with Python but still have basic coding knowledge

import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import warnings


warnings.simplefilter('ignore', category=FutureWarning)


# Load a binary blob of specified data type (defaults to float32)
def load_binary(filename, **kwargs):
    data_type = kwargs.pop('data_type', np.float32)
    with open(filename, 'rb') as fp:
        data = np.fromfile(fp, dtype=data_type)
    return data


# A bit of sugar coating for easy loading of data from files
def load_dataset(dataset='training'):
    import tensorflow as tf
    tf.random.set_seed(9)
    np.random.seed(19)  # For reproducibility purposes
    if dataset not in ['training', 'validate']:
        print('Error: dataset %s not valid. Only valid values = [training, validate]' % dataset)
        raise ValueError
    with open('data/%s_labels.csv' % dataset, 'r') as fp:
        labels = [line.replace('\n', '').split(',') for line in fp.readlines()[1:]]
    np.random.shuffle(labels)
    x, y = [], []
    for label in labels:
        filepath = 'data/signals/%s/%s' % (dataset, label[0])
        x.append(load_binary(filepath))
        y.append(label[1])
    return x, y


# Converts train type to 1-hot encoded vector, treating unknown as it's own type
def train_to_id5(train_type):
    from keras.utils import to_categorical
    tt_lut = {'train_a': 0, 'train_b': 1, 'train_c': 2, 'train_d': 3, 'unknown': 4}
    return to_categorical(tt_lut[train_type], 5)


# convert train type to 1-hot encoded vector, treating unknown as lack of a type
def train_to_id4(train_type):
    tt_lut = {
        'train_a': [1, 0, 0, 0],
        'train_b': [0, 1, 0, 0],
        'train_c': [0, 0, 1, 0],
        'train_d': [0, 0, 0, 1],
        'unknown': [0, 0, 0, 0]
    }
    return tt_lut[train_type]


def plot_validation_history(logger, acc=None):
    val_acc = logger.history['val_accuracy'][-1]
    if acc is None:
        acc = val_acc
    epochs = len(logger.history['val_accuracy'])
    plt.title('Accuracy over epochs')
    plt.plot(logger.history['val_accuracy'])
    plt.plot([0, epochs], [val_acc, val_acc], ls='--')
    plt.legend(['validation accuracy history', 'validation accuracy = %.2f%%' % (100. * acc)])
    plt.xlabel('Epochs')
    plt.ylim(0.0, 1.0)
    plt.tight_layout()
    plt.show()


# A bit of syntax sugar for easy adjustment of plot sizes
def plot_size(width, height):
    matplotlib.rcParams['figure.figsize'] = width, height
