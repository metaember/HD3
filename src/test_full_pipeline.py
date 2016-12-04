import loaders as ld
import formatters as fmt
import transformers as trf
import functions as fct
import classifiers as cl

import time

#
# Loading and formatting data
#
data_loaded = ld.load_data("mnist")
data_formatted = fmt.format_dataset(data_loaded)

#
# Applying transformations to data
#
t_drop = trf.Transformer("drop")
data_transformed_drop = t_drop.transform(data_formatted, 500)

t_G = trf.Transformer("g")
data_transformed_G = t_G.transform(data_formatted, 500)

t_Gcirc = trf.Transformer("g_circ")
data_transformed_Gcirc = t_Gcirc.transform(data_formatted, 500)

# t_HD3 = trf.Transformer("hd3")
# data_transformed_HD3 = t_HD3.transform(data_formatted, target_dimension)

#
# Applying functions to data
#
f_sigmoid = fct.Functions("sigmoid")
data_transformed_G_function_sigmoid = f_sigmoid.apply(data_transformed_G)

#
# Applying classifiers to data, returns accuracy and time
# (we should also take into consideration the time to apply the transformations)

# classic NN with softmax, using TensorFlow
c_softmax = cl.Classifier("softmax", "tf")
start_standard_softmax = time.clock()
accuracy_standard_softmax = c_softmax.classify(data_formatted, 0.8, 0.2, 100, 0.2)
end_standard_softmax = time.clock()
print("It took {} seconds".format(round(end_standard_softmax - start_standard_softmax, 2)))
print("Accuracy is {} with {} dimensions".format(accuracy_standard_softmax, data_formatted[0].shape[1]))