import loaders as ld
import formatters as fmt
import transformers as trf
import functions as fct
import classifiers as cl

import time

# Instantiate transformers
t_drop = trf.Transformer("drop")
t_G = trf.Transformer("g")
t_Gcirc = trf.Transformer("g_circ")

# Instantiate functions
f_identity = fct.Functions("identity")
f_sigmoid = fct.Functions("sigmoid")
f_sigmoid_prime = fct.Functions("sigmoid_prime")
f_signum = fct.Functions("signum")
f_tanh = fct.Functions("tanh")
f_softmax = fct.Functions("softmax")

# Instantiate classifiers
c_softmax_tf = cl.Classifier("softmax", "tf")



# MNIST

# Load and format data
data_loaded = ld.load_data("mnist")
data_formatted = fmt.format_dataset(data_loaded)


# Pipeline 0: benchmark (softmax classifier, tensorFlow)
start_benchmark = time.clock()
accuracy_benchmark = c_softmax_tf.classify(data_formatted, 0.8, 0.2, 100, 0.2)
end_benchmark = time.clock()
print("It took {} seconds".format(round(end_benchmark - start_benchmark, 2)))
print("Accuracy is {} with {} dimensions".format(accuracy_benchmark, data_formatted[0].shape[1]))


# Pipeline 1: drop 284 components, apply identity, classify using softmax NN
data_transformed_drop = t_drop.transform(data_formatted, 500)
data_transformed_drop_function_identity = f_identity.apply(data_transformed_drop)

start_t_drop_f_identity_c_softmax = time.clock()
accuracy_t_drop_f_identity_c_softmax = c_softmax_tf.classify(data_transformed_drop_function_identity, 0.8, 0.2, 100, 0.2)
end_t_drop_f_identity_c_softmax = time.clock()
print("It took {} seconds".format(round(end_t_drop_f_identity_c_softmax - start_t_drop_f_identity_c_softmax, 2)))
print("Accuracy is {} with {} dimensions".format(accuracy_t_drop_f_identity_c_softmax, data_transformed_drop_function_identity[0].shape[1]))