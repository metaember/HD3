import loaders as ld
import formatters as fmt
import transformers as trf
import functions as fct
import plotters as plts
import classifiers as cl

#
# Loading and formatting data
#
data_loaded = ld.load_data("mnist")
data_formatted = fmt.format_dataset(data_loaded)
data_normalized = fmt.normalize_dataset(data_formatted)
print(data_normalized[0].shape)

#
# Applying transformations to data
#
t_drop = trf.Transformer("drop")
data_transformed_drop = t_drop.transform(data_formatted, 700)

t_G = trf.Transformer("g")
data_transformed_G = fmt.normalize_dataset(t_G.transform(fmt.normalize_dataset(data_formatted), 500))

t_Gcirc = trf.Transformer("g_circ")
data_transformed_Gcirc = t_Gcirc.transform(data_formatted, 500)

t_HD3 = trf.Transformer("hd3")
data_transformed_HD3 = t_HD3.transform(data_formatted, 500)

#
# Applying functions to data
#
f_sigmoid = fct.Functions("sigmoid")
data_transformed_G_function_sigmoid = f_sigmoid.apply(data_transformed_G)

#
# Plotting data
#
p_euclidian = plts.Plotter("euclidian", 1000)

# p_euclidian.plot(data_transformed_drop, data_formatted)
p_euclidian.plot(data_transformed_G, data_formatted)
p_euclidian.plot(data_transformed_Gcirc, data_formatted)
# p_euclidian.plot(data_transformed_HD3, data_formatted)