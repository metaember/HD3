# User interactions
import argparse

parser = argparse.ArgumentParser(description="Compare speed and accuracy of NN's using various preprocessing methods.")
parser.add_argument("mode", help="Neural Net mode: 'softmax' or 'cnn' ?")
parser.add_argument("-p", "--preprocess", help="How to preprocess the data? Default: nothing")
parser.add_argument("-c", "--count", help = "How many sample pairs to use for the histograms? Default = 100")
parser.add_argument("-d", "--dataset", help="What dataset to use? Default: mnist")
parser.add_argument("-v", "--verbose", action="store_true",
                    help="increase output verbosity")
args = parser.parse_args()

# constants
MODES = ("none", "softmax", "cnn")
PREPROC = (None,"G")
DATASETS = ("mnist",)

if args.dataset is None:
    args.dataset = "mnist"

if args.count is None:
    args.count = 100

# main UI switch
if args.mode not in MODES:
    raise NameError("{} is not an available mode. Try {}".format(args.mode, MODES))
elif args.preprocess not in PREPROC:
    raise NameError("{} is not an available preprocessing option. Try {}".format(args.preprocess, PREPROC))
elif args.dataset not in DATASETS:
    raise NameError("{} is not an available dataset. Try {}".format(args.dataset, DATASETS))
else:

    if args.mode is not "none":
        from test_full_pipeline import *
        print("Starting. Mode = {}, preprocessing = {}, dataset = {}, verbose = {}".format(
                    args.mode, args.preprocess, args.dataset, args.verbose))
        print()
        run_test(args.mode, preproc = args.preprocess, verbose=args.verbose)
    else:
        import preprocess
        preprocess.do()
