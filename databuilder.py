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

    p.add('-b', '--blacklist_character', required=True, type=str, metavar='STRING',
          help="In the user-provided input file, words that begin with this character will be ignored.")

    p.add('-i', '--raw_sentences', required=True, type=str, metavar="FILENAME",
          help="Input file containing raw corpus in plain-text format")

    p.add('-i', '--segmented_sentences', required=True, type=str, metavar="FILENAME",
          help="Input file containing segmented corpus in plain-text format")

    p.add('-o', '--output_file', required=True, type=str, metavar="FILENAME",
          help="Output file where data will be saved")

    return p.parse_args(args=arguments)

class DataBuilder (object):

    def __init__(self, *,
                 tokenizer: Tokenizer,
                 blacklist_char: str,
                 raw_sentences: Iterable[str],
                 segmented_sentences: Iterable[str],
                 data_set: str,
                 output_file: ??????):

        # plan: for loop over zip(raw_sentences, segmented_sentences). for loop of words in sentences. throw out *words. write to file.
        data_pairs = open("data_pairs" + data_set + ".txt", "w+") # create a file to store the raw and segmented words in pairs
        processed_words = set([])

        for sent1, sent2 in zip(raw_sentences, segmented_sentences):
            for word1, word2 in zip(sent1, sent2):
                if word2.startswith(blacklist_char) or word1 in processed_words:
                    continue
                # keep writing here
                data_pairs.write(word1, "\t", word2, "\n") # raw then segmented with tab between.
                                                           # each pair should be on a separate line
                processed_words.add(word1) # make sure the processed list registers the new processed word

        data_pairs.close()



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