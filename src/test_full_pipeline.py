import loaders as ld
import formatters as fmt
import transformers as trf
import functions as fct
import classifiers as cl

import time




def run_test(mode, preproc, verbose):

    if verbose:
        print("running, mode {}, proproc {}, verbose {}".format(mode,preproc,verbose))

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


    if preproc is None:
        pass
    elif preproc == "G":
        data_formatted = data_transformed_G
    else:
        raise NotImplemented("Not an implemented transformation")

    #
    # Applying classifiers to data, returns accuracy and time
    # (we should also take into consideration the time to apply the transformations)
    #


    if mode == "softmax":
        # classic NN with softmax, using TensorFlow
        classif = cl.Classifier("softmax", "tf")
    elif mode == "cnn":
        # CNN using TensorFlow
        classif = cl.Classifier("cnn", "tf")
    else:
        raise NameError("Mode must be 'softmax' or 'cnn'")


    start_timing = time.clock(), time.time()
    accuracy = classif.classify(data_formatted, 0.8, 0.2, 50, 1e-4, 2000, verbose=verbose)
    end_timing = time.clock(), time.time()

    wall_time = end_timing[1] - start_timing[1]
    proc_time = end_timing[0] - start_timing[0]

    short_accuracy = (int(accuracy * 10000) / 100)
    print("Training using {} took {}sec ({}min) of wall time and {}sec ({}min) of processor time".format(
        mode,round(wall_time, 2), round(wall_time/60,2), round(proc_time, 2), round(proc_time/60,2)))
    print("Accuracy is {} with {} dimensions".format(short_accuracy, data_formatted[0].shape[1]))


if __name__ == "__main__":
    print("Warning: Deprecated. You called 'test_full_pipeline.py' directly. Next time, call it using HD3.py")
    print()
    # params
    mode = "cnn" # cnn or softmax
    run_test(mode,preproc = None, verbose=True)
