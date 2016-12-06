import classifiers_tf as cl_tf
# import classifiers_torch as cl_torch
# import classifiers_theano as cl_theano

class Classifier:
    """
    Mainly TensorFlow code to classify data
    """
    def __init__(self, type, framework):
        self.type = type
        self.framework = framework

        def classify(self, dataset, percentage_train, percentage_test, batch_size, learning_rate):

            if self.type == "softmax":
                if self.framework == "tf":
                    return cl_tf.classify_with_softmax_nn(dataset, percentage_train, percentage_test, batch_size, learning_rate)
                elif self.framework == "torch":
                    raise NotImplemented('Torch implementation not done yet')
                else:
                    raise NameError("Framework must be 'tf' or 'torch'")


            elif self.type == "cnn":
                if self.framework == "tf":
                    return cl_tf.classify_with_conv_nn(dataset, percentage_train, percentage_test, batch_size, learning_rate)
                elif self.framework == "torch":
                    raise NotImplemented('Torch implementation not done yet')
                else:
                    raise NameError("Framework must be 'tf' or 'torch'")

            else:
                raise NameError("Type must be 'softmax' or 'cnn'")
