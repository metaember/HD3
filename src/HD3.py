# User interactions
import argparse

parser = argparse.ArgumentParser(description="Compare speed and accuracy of NN's using various preprocessing methods.")
parser.add_argument("mode", help="Neural Net mode: 'softmax' or 'cnn' ?")
parser.add_argument("-p", "--preprocess", help="How to preprocess the data? Default: nothing")
parser.add_argument("-v", "--verbose", action="store_true",
                    help="increase output verbosity")
args = parser.parse_args()


MODES = ("softmax", "cnn")

if args.mode not in MODES:
    raise NameError("{} is not an available mode. Try {}".format(args.mode, MODES))
else:
    from test_full_pipeline import *
    print("Starting. Mode = {}, preprocessing = {}, verbose = {}".format(args.mode, args.preprocess, args.verbose))
    print()
    run_test(args.mode, verbose=args.verbose)
