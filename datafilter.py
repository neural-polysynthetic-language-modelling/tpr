import argparse
import configargparse
from typing import Iterable, List, MutableSet, Optional
from tokenizer import *

# This file is for building training data for morpheme seq2seq segmenter.

def configure(arguments: List[str]) -> argparse.Namespace:

    p = configargparse.get_argument_parser(description="Build training data for seq2seq segmenter")
    p.add('-c', '--config', required=False, is_config_file=True, type=str, metavar='FILENAME',
          help='configuration file')

    p.add('--tokenizer', required=True, type=str, metavar='FILENAME',
          help='Pickle file containing a Tokenizer object')

    p.add('-i', '--word_pairs', required=True, type=str, metavar="FILENAME",
          help="Input file containing word pairs in plain-text format")

    p.add('-o', '--train', required=True, type=str, metavar="FILENAME",
          help="Output file where training set will be saved")

    p.add('-o', '--dev',required=True, type=str, metavar="FILENAME",
          help="Output file where development set will be saved")

    p.add('-o', '--test', required=True, type=str, metavar="FILENAME",
          help="Output file where test set will be saved")

    return p.parse_args(args=arguments)


class DataFilter (object):

    def __init__(self, *,
                 tokenizer: Tokenizer,
                 blacklist_char: str,
                 segmented_sentences: Iterable[str],
                 output_file: ??????):

        # plan: for loop over zip(raw_sentences, segmented_sentences). for loop of words in sentences. throw out *words. write to file.
        #pairs = open("data_pairs.txt", "r") # open file to sort out the duplicate words for dev and test
        #pairs.readlines()
        # TODO: rewrite so only accepts two files. remove for loop??
        train_data = open("data_pairs_train.txt", "r")
        dev_data = open("data_pairs_dev.txt", "r")
        test_data = open("data_pairs_test.txt", "r")

        train_data.readlines()
        dev_data.readlines()
        test_data.readlines()

        dev_data_noDup = open("data_pairs_dev_noDuplicates.txt", "w+")
        test_data_noDup = open("data_pairs_test_noDuplicates.txt", "w+")


        for line in dev_data:
            if line in train_data:  # if we've already seen the word, move on
                continue
            dev_data_noDup.write(line)

        for line in test_data:
            if line in dev_data:
                continue
            test_data_noDup.write(line)

        train_data.close()
        dev_data.close()
        test_data.close()

        dev_data_noDup.close()
        test_data_noDup.close()



def main(args: argparse.Namespace) -> None:

    import pickle
    # raw_sentences is the entire dataset
    with open(args.raw_sentences, 'rt') as raw_sentences, \
            open(args.segmented_sentences, 'rt') as segmented_sentences, \
            open(args.output_file, 'wb') as output_file:

        db = DataBuilder(tokenizer=Tokenizer.load(args.tokenizer),
                                blacklist_char=args.blacklist_character,
                                raw_sentences=raw_sentences,
                                segmented_sentences=segmented_sentences,
                                output_file=output_file)


if __name__ == "__main__":

    import sys

    main(configure(arguments=sys.argv[1:]))