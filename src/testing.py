import loaders as ld
import formatters as fmt
import transformers as trf
import functions as fct
import classifiers as cl


data_loaded = ld.load_data("mnist")
data_formatted = fmt.format_dataset(data_loaded)

t_G = trf.Transformer("g")
data_transformed_G = t_G.transform(data_formatted, 400)

print(data_transformed_G[0].shape)
