import gzip
import tarfile
import pickle
import os
import numpy as np


def load_data(dataset, batch = None):
    sets = ("mnist", "cifar10", "cifar100")
    dataset = dataset.lower()
    if dataset not in sets:
        raise ValueError("Available datasets are : {}".format(", ".join(sets)))

    check_dataset(dataset)

    if dataset.startswith("cifar"):
        assert batch is not None and batch in range(6)

    if dataset == "mnist":
        return load_data_mnist()
    elif dataset == "cifar10":
        return load_data_cifar(10, batch)[b'data']
    elif dataset == "cifar100":
        return load_data_cifar(100, batch)[b'data']
    else:
        assert False


#
#  Code for Mnist
#

def load_data_mnist():
    """Return the MNIST data as a tuple containing the training data,
    the validation data, and the test data.

    The ``training_data`` is returned as a tuple with two entries.
    The first entry contains the actual training images.  This is a
    numpy ndarray with 50,000 entries.  Each entry is, in turn, a
    numpy ndarray with 784 values, representing the 28 * 28 = 784
    pixels in a single MNIST image.

    The second entry in the ``training_data`` tuple is a numpy ndarray
    containing 50,000 entries.  Those entries are just the digit
    values (0...9) for the corresponding images contained in the first
    entry of the tuple.

    The ``validation_data`` and ``test_data`` are similar, except
    each contains only 10,000 images.

    This is a nice data format, but for use in neural networks it's
    helpful to modify the format of the ``training_data`` a little.
    That's done in the wrapper function ``load_data_wrapper_mnist()``, see
    below.
    """
    with gzip.open('../data/mnist.pkl.gz', 'rb') as f:
        training_data, validation_data, test_data = pickle.load(f, encoding='bytes')
    return (training_data, validation_data, test_data)


def load_data_wrapper_mnist():
    """Return a tuple containing ``(training_data, validation_data,
    test_data)``. Based on ``load_data_mnist``, but the format is more
    convenient for use in our implementation of neural networks.

    In particular, ``training_data`` is a list containing 50,000
    2-tuples ``(x, y)``.  ``x`` is a 784-dimensional numpy.ndarray
    containing the input image.  ``y`` is a 10-dimensional
    numpy.ndarray representing the unit vector corresponding to the
    correct digit for ``x``.

    ``validation_data`` and ``test_data`` are lists containing 10,000
    2-tuples ``(x, y)``.  In each case, ``x`` is a 784-dimensional
    numpy.ndarry containing the input image, and ``y`` is the
    corresponding classification, i.e., the digit values (integers)
    corresponding to ``x``.

    Obviously, this means we're using slightly different formats for
    the training data and the validation / test data.  These formats
    turn out to be the most convenient for use in our neural network
    code."""
    tr_d, va_d, te_d = load_data_mnist()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result_mnist(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    return (training_data, validation_data, test_data)

def vectorized_result_mnist(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e






#
#  Stuff for CIFAR
#


def load_data_cifar(cifar_version, batch):
    """
    five batches labled 1-5, and a test batch labled 0

    Details for cifar 10 version:

        Loaded in this way, each of the batch files contains a dictionary with the following elements:

        data -- a 10000x3072 numpy array of uint8s. Each row of the array stores a 32x32 colour image.
            The first 1024 entries contain the red channel values, the next 1024 the green, and the final
            1024 the blue. The image is stored in row-major order, so that the first 32 entries of the array
            are the red channel values of the first row of the image.

        labels -- a list of 10000 numbers in the range 0-9. The number at index i indicates the label
            of the ith image in the array data.


        The dataset contains another file, called batches.meta. It too contains a Python dictionary object.
        It has the following entries:

        label_names -- a 10-element list which gives meaningful names to the numeric labels in the labels array
            described above. For example, label_names[0] == "airplane", label_names[1] == "automobile", etc.

    """
    cifar_version = int(cifar_version)
    if cifar_version not in (10,100):
        raise ValueError("Cifar version must be 10 or 100")


    # Check if the data folder is present
    if not os.path.exists("data"):
        os.makedirs("data")

    # Extract
    print("cwd: ", os.getcwd())
    tar = tarfile.open("data/cifar-{}-python.tar.gz".format(cifar_version))
    file_names = tar.getnames()
    for file_name in file_names:
        tar.extract(file_name,"data/")
    tar.close()

    #unpickle

    batch  = int(batch)
    if batch not in range(6):
        raise ValueError("Batch must be an int between 0 and 5")

    if batch == 0:
        return unpickle("data/cifar-{}-batches-py/test_batch".format(cifar_version))
    else:
        return unpickle("data/cifar-{}-batches-py/data_batch_{}".format(cifar_version, batch))


def unpickle(filename):

    with open(filename, "rb") as f:
        dictionary = pickle.load(f, encoding='bytes')

    return dictionary





#
# conveniance
#

def check_dataset(dataset):
    """
    Check if dataset is in the data directory. If not, will download it.
    """
    dataset_names = dict()
    dataset_names["mnist"] = "mnist.pkl.gz"
    dataset_names["cifar10"] = "cifar-10-python.tar.gz"
    dataset_names["cifar100"] = "cifar-100-python.tar.gz"

    new_path = os.path.join(
        os.path.split(__file__)[0],
        "..",
        "data",
        dataset_names[dataset]
    )
    f_name = os.path.join(
        os.path.split(__file__)[0],
        "..",
        "data"
    )
    print(new_path)
    if (not os.path.isfile(new_path)):
        from six.moves import urllib
        if dataset in ("cifar10", "cifar100"):
            origin = 'https://www.cs.toronto.edu/~kriz/' + dataset_names[dataset]
        else:
            origin = "http://deeplearning.net/data/mnist/mnist.pkl.gz"

        print('Downloading data from {}'.format(origin))
        urllib.request.urlretrieve(origin, new_path)
