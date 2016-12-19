import loaders as ld
import formatters as fmt
import transformers as trf
import functions as fct
import plotters as plts
# import classifiers as cl

'''
Load and format
'''
d_loaded = ld.load_data("mnist")
d_formatted = fmt.format_dataset(d_loaded)


'''
Apply transform and functions
'''
t_drop = trf.Transformer("drop")
t_G = trf.Transformer("g")
t_Gcirc = trf.Transformer("g_circ")
t_HD3 = trf.Transformer("hd3")

# ts = [t_drop, t_G, t_Gcirc, t_HD3]
ts = [t_drop, t_G]
t_dict = ["Drop", "Gaussian", "Gauss Circ", "HD3"]

target_dimension_1 = 600
target_dimension_2 = 500
target_dimension_3 = 400

# tds = [target_dimension_1, target_dimension_2, target_dimension_3]
tds = [target_dimension_1, target_dimension_2]

f_identity = fct.Function("identity")
f_sigmoid = fct.Function("sigmoid")
f_sigmoid_prime = fct.Function("sigmoid_prime")
f_signum = fct.Function("signum")
f_tanh = fct.Function("tanh")
f_softmax = fct.Function("softmax")

# fs = [f_identity, f_sigmoid, f_sigmoid_prime, f_signum, f_tanh, f_softmax]
fs = [f_identity, f_sigmoid]
f_dict = ["Identity", "sigmoid"]


'''
Compute combinations of TRANSFORM/TARGET_DIM/FUNCTION to yield various datasets stored in a 3D array:

i,j,k -> dataset subject to transform i, target_dim j, function k
according to the arrays defined above

Each dataset is a 2uple containing images and labels ready ready to be classified

The "dict" arrays are just used to add a legend to the plots
'''
dataset_transform_targetdim_function = np.ndarray(shape=(len(ts),len(tds),len(fs)), dtype=object)

for i in range(len(ts)):
    for j in range(len(tds)):
        for k in range(len(fs)):
            a = fs[k].apply(ts[i].transform(d_formatted, tds[j]))
            print(a[0].shape)
            dataset_transform_targetdim_function[i][j][k] = a

print(dataset_transform_targetdim_function.shape)


'''
Plot data
'''
pairs_of_point = plts.generate_pairs_points(d_formatted, 1000)

p_euclidian = plts.Plotter("euclidian", pairs_of_point)
p_euclidian.superplot(dataset_transform_targetdim_function, d_formatted, -1, -1, -1, t_dict, tds, f_dict)

# p_angular = plts.Plotter("angular", pairs_of_point)