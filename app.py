import argparse
import sys

import classifier
import generator
import trainer


def _run_module(module_main, argv, name):
    sys.argv = [name] + argv
    module_main()


def main():
    parser = argparse.ArgumentParser(description="Tetris classifier app")
    subparsers = parser.add_subparsers(dest="command", required=True)

    gen_parser = subparsers.add_parser("generate", help="Run descriptor generator")
    gen_parser.add_argument("args", nargs=argparse.REMAINDER)

    train_parser = subparsers.add_parser("train", help="Run model trainer")
    train_parser.add_argument("args", nargs=argparse.REMAINDER)

    cls_parser = subparsers.add_parser("classify", help="Run classifier")
    cls_parser.add_argument("args", nargs=argparse.REMAINDER)

    args = parser.parse_args()

    if args.command == "generate":
        _run_module(generator.main, args.args, "generator.py")
    elif args.command == "train":
        _run_module(trainer.main, args.args, "trainer.py")
    elif args.command == "classify":
        _run_module(classifier.main, args.args, "classifier.py")


if __name__ == "__main__":
    main()
