import loaders as ld
import formatters as fmt
import transformers as trf
import functions as fct
import classifiers as cl


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

t_HD3 = trf.Transformer("hd3")
data_transformed_HD3 = t_HD3.transform(data_formatted, 500)

print(data_transformed_HD3[0].shape, data_transformed_HD3[1].shape)