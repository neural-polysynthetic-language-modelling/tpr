import argparse
import configargparse
from typing import Iterable, List
import difflib

# This file is for building training data for morpheme seq2seq segmenter.

def configure(arguments: List[str]) -> argparse.Namespace:

    p = configargparse.get_argument_parser(description="Build training data for seq2seq segmenter")
    p.add('-c', '--config', required=False, is_config_file=True, type=str, metavar='FILENAME',
          help='configuration file')

    p.add('-i', '--train_pairs', required=True, type=str, metavar="FILENAME",
          help="Input file containing training word pairs in plain-text format")

    p.add('-i', '--dev_pairs', required=True, type=str, metavar="FILENAME",
          help="Input file containing dev word pairs in plain-text format")

    p.add('-i', '--test_pairs', required=True, type=str, metavar="FILENAME",
          help="Input file containing test word pairs in plain-text format")

    p.add('-o', '--dev_output_file',required=True, type=str, metavar="FILENAME",
          help="Output file where development set will be saved")

    p.add('-o', '--test_output_file', required=True, type=str, metavar="FILENAME",
          help="Output file where test set will be saved")

    return p.parse_args(args=arguments)


class DataFilter (object):

    def __init__(self, *,
                 train_pairs: Iterable[str],
                 dev_pairs: Iterable[str],
                 test_pairs: Iterable[str],
                 dev_output_file: str,
                 test_output_file: str):


        training_words = set([])
        for line in train_pairs:
            training_words.add (line.strip().split('\t')[0])

        # Remove all words in dev that are in train
        dev_words = set([])
        with open(dev_output_file, "w+") as dev_output:
            for line in dev_pairs:
                word = line.strip().split('\t')[0]
                if word not in training_words:
                    dev_output.write(line)
                    dev_words.add(word)

        # Remove all words in test that are in dev
        with open(test_output_file, "w+") as test_output:
            for line in test_pairs:
                word = line.strip().split('\t')[0]
                if word not in dev_words:
                    test_output.write(line)


def main(args: argparse.Namespace) -> None:

    with open(args.train_pairs, 'rt') as train_pairs, \
            open(args.dev_pairs, 'rt') as dev_pairs, \
            open(args.test_pairs, 'rt') as test_pairs:
        db = DataFilter(train_pairs=train_pairs,
                        dev_pairs=dev_pairs,
                        test_pairs=test_pairs,
                         dev_output_file=args.dev_output_file,
                        test_output_file=args.test_output_file)

if __name__ == "__main__":

    import sys

    main(configure(arguments=sys.argv[1:]))