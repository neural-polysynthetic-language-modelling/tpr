import argparse
import configargparse
from typing import Iterable, List, MutableSet, Optional
from tokenizer import *
import difflib

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
                 file1name: str,
                 file2name: str,
                 output_file: ??????):


        file1 = "data_pairs_" + file1name + ".txt"
        with open(file1) as f1:
            f1_text = f1.readlines()

        file2 = "data_pairs_" + file2name + ".txt"
        with open(file2) as f2:
            f2_text = f2.readlines()

        file2_noDuplicates = open("data_filtered_" + file2name + ".txt", "w+")

        # Find and write the diff to a file
        for line in difflib.unified_diff(f1_text, f2_text, fromfile=file1, tofile=file2, lineterm=''):
            if line.startswith("+") and line.startswith("+++") != True: # if it is diff, write to file
                file2_noDuplicates.write(line[1:]) # don't include the +

        file2_noDuplicates.close()





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