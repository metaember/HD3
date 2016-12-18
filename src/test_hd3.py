import loaders as ld
import formatters as fmt
import transformers as trf
import functions as fct
import plotters as plts
import classifiers as cl


'''
One liner debug
'''
import numpy as np

# d_f = np.apply_along_axis(np.linalg.norm, 1, data_formatted[0])
# print(np.amin(d_f), np.amax(d_f))

'''
Load and format
'''
d_loaded = ld.load_data("mnist")
d_formatted = fmt.format_dataset(d_loaded)
# /!\ we shouldn't normalize when plotting !!
d_N = fmt.normalize_dataset(d_formatted)

'''
Apply transform
'''
t_drop = trf.Transformer("drop")
t_G = trf.Transformer("g")
t_Gcirc = trf.Transformer("g_circ")
t_HD3 = trf.Transformer("hd3")

f_identity = fct.Function("identity")
f_sigmoid = fct.Function("sigmoid")
f_sigmoid_prime = fct.Function("sigmoid_prime")
f_signum = fct.Function("signum")
f_tanh = fct.Function("tanh")
f_softmax = fct.Function("softmax")

d_drop = t_drop.transform(d_formatted, 500)
d_drop_identity = f_identity.apply(d_drop)

d_G = t_G.transform(d_formatted, 500)
d_G_identity = f_identity.apply(d_G)

d_Gcirc_identity = fmt.normalize_dataset(t_Gcirc.transform(d_N, 500))

d_HD3_identity = fmt.normalize_dataset(t_HD3.transform(d_N, 500))

# '''
# Apply function
# '''
# f_sigmoid = fct.Functions("sigmoid")
# data_transformed_G_function_sigmoid = f_sigmoid.apply(data_transformed_G)
#
# '''
# Plot data
# '''

# we should create a seed if we want to compare plots, otherwise we will plot the quotient of distance of different couple of points, which introduce a new non deterministic parameter

pairs_of_point = plts.generate_pairs_points(d_formatted, 1000)

p_euclidian = plts.Plotter("euclidian", pairs_of_point)

# d_drop_N = fmt.normalize_dataset(d_drop)
# d_drop_identity_N = fmt.normalize_dataset(d_drop_identity)

p_euclidian.plot(fmt.normalize_dataset(d_G), d_formatted)
p_euclidian.plot(fmt.normalize_dataset(d_G_identity), fmt.normalize_dataset(d_formatted))

# p_euclidian.plot(d_drop_identity, d_normalized)
# p_euclidian.plot(d_G_identity, d_normalized)
# p_euclidian.plot(d_Gcirc_identity, d_normalized)
# p_euclidian.plot(d_HD3_identity, d_normalized)