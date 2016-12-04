''' Not doing something clean because we
    can't really code a generic solution,
    this step should be integrated to loaders.py
    to present data from every dataset in the
    same format, that is :

    dataset is a 2 uple
    dataset[0] = inputs (in R_nbInput,dimensionInput)
    dataset[1] = labels (in R_nbInput,dimensionOutput)

    Allows to choose ourself the percentage for training, testing later
    (as coded in classifiers : percentage_training, percentage_test are
    both arguments of the implemented methods)

    Example : MNIST
              dataset[0] in R_80000,784
              dataset[1] in R_80000,10

    Note : Vectorised labels so it is easier to plug the dataset into
           TensorFlow NN

'''
import numpy as np

vectorise_mnist_labels = lambda i: [1 if j==i else 0 for j in range(10)]

def format_dataset(dataset):

    train = dataset[0]
    test = dataset[1]
    # validation = dataset[2]
    # not sure there are labels in the validation dataset

    formatted_dataset_input = np.concatenate((train[0], test[0]), axis=0)
    formatted_datased_labels = np.concatenate((train[1], test[1]), axis=0)
    formatted_dataset_vectorised_labels = np.array(list(map(vectorise_mnist_labels, formatted_datased_labels)))

    return (formatted_dataset_input, formatted_dataset_vectorised_labels)