# User interactions
import argparse

parser = argparse.ArgumentParser(description="Compare speed and accuracy of NN's using various preprocessing methods.")
parser.add_argument("mode", help="Neural Net mode: 'softmax' or 'cnn' ?")
parser.add_argument("-p", "--preprocess", help="How to preprocess the data? Default: nothing")
parser.add_argument("-d", "--dataset", help="What dataset to use? Default: mnist")
parser.add_argument("-v", "--verbose", action="store_true",
                    help="increase output verbosity")
args = parser.parse_args()

# constants
MODES = ("softmax", "cnn")
PREPROC = tuple()
DATASETS = ("mnist",)

# main UI switch
if args.mode not in MODES:
    raise NameError("{} is not an available mode. Try {}".format(args.mode, MODES))
elif args.preprocess not in PREPROC:
    raise NameError("{} is not an available preprocessing option. Try {}".format(args.preprocess, PREPROC))
elif args.dataset.tolower() not in DATASETS:
    raise NameError("{} is not an available dataset. Try {}".format(args.dataset, DATASETS))
else:
    from test_full_pipeline import *
    print("Starting. Mode = {}, preprocessing = {}, dataset = {}, verbose = {}".format(
                args.mode, args.preprocess, args.dataset, args.verbose))
    print()
    run_test(args.mode, verbose=args.verbose)
