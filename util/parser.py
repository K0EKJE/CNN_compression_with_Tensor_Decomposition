import argparse
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument("--train", dest="train", action="store_true")
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--resume_d', action='store_true',
                    help='resume from decomposed checkpoint')


parser.add_argument("--decompose", dest="decompose", action="store_true")
# the following three only used for decomposition case
parser.add_argument("--add", dest="add", action="store_true")
parser.add_argument("--fine_tune", dest="fine_tune", action="store_true")
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate for \
fine tuning decomposed model')


parser.add_argument("--run_model", dest="run_model", action="store_true")

parser.add_argument("--tucker", dest="tucker", action="store_true", \
    help="Use tucker decomposition. uses cp by default")

parser.set_defaults(train=False)
parser.set_defaults(decompose=False)
parser.set_defaults(add=False)
parser.set_defaults(fine_tune=False)
parser.set_defaults(run_model=False)

parser.set_defaults(tucker=False)

args = parser.parse_args()
