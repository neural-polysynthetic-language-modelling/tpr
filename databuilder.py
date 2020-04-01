import argparse
import configargparse
from typing import Iterable, List, MutableSet, Optional
import pdb


# This file is for building training data for morpheme seq2seq segmenter.

def configure(arguments: List[str]) -> argparse.Namespace:
    p = configargparse.get_argument_parser(description="Build data for seq2seq segmenter")
    p.add('-c', '--config', required=False, is_config_file=True, type=str, metavar='FILENAME',
          help='configuration file')

    p.add('-b', '--blacklist_character', required=True, type=str, metavar='STRING',
          help="In the user-provided input file, words that begin with this character will be ignored.")

    p.add('-i', '--raw_sentences', required=True, type=str, metavar="FILENAME",
          help="Input file containing raw corpus in plain-text format")

    p.add('-i', '--segmented_sentences', required=True, type=str, metavar="FILENAME",
          help="Input file containing segmented corpus in plain-text format")

    p.add('-o', '--output_file', required=True, type=str, metavar="FILENAME",
          help="Output file where data will be saved")

    return p.parse_args(args=arguments)


class DataBuilder(object):

    def __init__(self, *,
                 blacklist_char: str,
                 raw_sentences: Iterable[str],
                 segmented_sentences: Iterable[str],
                 output_file: str):

        processed_words = set([])

        with open(output_file, "w+") as data_pairs: # create a file to store the raw and segmented words in pairs

            for sent1, sent2 in zip(raw_sentences, segmented_sentences):
                sent1_split = sent1.split (' ')
                sent2_split = sent2.split (' ')
                assert(len(sent1_split) == len (sent2_split))
                for word1, word2 in zip(sent1_split, sent2_split):
                    if word2.startswith(blacklist_char) or word1 in processed_words:
                        continue
                    data_pairs.write(word1 + "\t" + word2 + "\n")  # raw then segmented with tab between.
                    processed_words.add(word1)  # make sure the processed list registers the new processed word


def main(args: argparse.Namespace) -> None:
    with open(args.raw_sentences, 'rt') as raw_sentences, \
            open(args.segmented_sentences, 'rt') as segmented_sentences:
        db = DataBuilder(blacklist_char=args.blacklist_character,
                         raw_sentences=raw_sentences,
                         segmented_sentences=segmented_sentences,
                         output_file=args.output_file)


if __name__ == "__main__":
    import sys

    main(configure(arguments=sys.argv[1:]))
