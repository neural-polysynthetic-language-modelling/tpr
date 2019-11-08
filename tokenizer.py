import argparse
import configargparse
from typing import List


class Tokenizer:

    def words(self, sentence: str) -> List[str]:
        raise NotImplementedError

    def morphemes(self, word: str) -> List[str]:
        raise NotImplementedError

    def graphemes(self, morpheme: str) -> List[str]:
        raise NotImplementedError

    @staticmethod
    def load(filename: str) -> 'Tokenizer':
        import pickle
        with open(filename, 'rb') as pickled_file:
            return pickle.load(pickled_file)


class MorphemeTokenizer(Tokenizer):

    def __init__(self, morpheme_delimiter: str, use_nltk_tokenizer=True):
        self.morpheme_delimiter = morpheme_delimiter
        self.use_nltk_tokenizer = use_nltk_tokenizer

    def words(self, sentence: str) -> List[str]:
        if self.use_nltk_tokenizer:
            from nltk.tokenize import word_tokenize
            return word_tokenize(text=sentence, preserve_line=True)
        else:
            return sentence.split()

    def morphemes(self, word: str) -> List[str]:
        return word.split(self.morpheme_delimiter)

    def graphemes(self, morpheme: str) -> List[str]:
        return list(morpheme)


class YupikMorphemeTokenizer(MorphemeTokenizer):

    grapheme_inventory = ['ngngw',
                          'ghhw', 'ngng', 'ghh*',
                          'ghh', 'ghw', 'ngw', 'gh*',
                          'gg', 'gh', 'kw', 'll', 'mm', 'ng', 'nn', 'qw', 'rr', 'wh',
                          'a', 'e', 'f', 'g', 'h', 'i', 'k', 'l', 'm', 'n', 'p', 'q',
                          'r', 's', 't', 'u', 'v', 'w', 'y', 'z']

    doubled_grapheme = {'l': 'll',
                        'r': 'rr',
                        'g': 'gg',
                        'gh*': 'ghh*',
                        'gh': 'ghh',
                        'ghw': 'ghhw',
                        'n': 'nn',
                        'm': 'mm',
                        'ng': 'ngng',
                        'ngw': 'ngngw'}

    def __init__(self, morpheme_delimiter: str, use_nltk_tokenizer=True):
        super().__init__(morpheme_delimiter=morpheme_delimiter, use_nltk_tokenizer=use_nltk_tokenizer)

    @staticmethod
    def double(grapheme: str, caps: str, index: int) -> str:

        doubled: str = YupikMorphemeTokenizer.doubled_grapheme[grapheme.lower()]

        if caps == 'ALL_CAPS':
            return doubled.upper()
        elif caps == 'lower':
            return doubled
        elif caps == 'Initial':
            if index == 0:
                return doubled[0].upper() + doubled[1:].lower()
            else:
                return doubled
        else:
            for i in range(len(grapheme)):
                if grapheme[i].isupper():
                    doubled[i] = doubled[i].upper()
            return doubled

    @staticmethod
    def to_list(yupik_string: str) -> List[str]:
        result = []
        end = len(yupik_string)

        while end > 0:

            found_grapheme = False

            for grapheme in YupikMorphemeTokenizer.grapheme_inventory:

                if yupik_string[:end].lower().endswith(grapheme):
                    result.insert(0, yupik_string[end - len(grapheme):end])
                    end -= len(grapheme)
                    found_grapheme = True
                    break

            if not found_grapheme:
                result.insert(0, yupik_string[end - 1:end])
                end -= 1

        return result

    @staticmethod
    def caps_status(string: str) -> str:
        if string == string.upper():
            caps = "ALL_CAPS"
        elif string == string.lower():
            caps = "lower"
        elif string == (string[0].upper() + string[1:].lower()):
            caps = "Initial"
        else:
            caps = "MiXEd_cAps"

        return caps

    def handle_analyzer_bug(self, graphemes: List[str]) -> str:
        used_fix = False
        result = list()
        for i in range(len(graphemes)):
            if i + 1 < len(graphemes) and (graphemes[i] == 'gh*' or graphemes[i] == 'ghh*') and graphemes[i + 1] != self.morpheme_delimiter:
                result.append(graphemes[i])
                result.append(self.morpheme_delimiter)
                used_fix = True
            elif graphemes[i] == '-':
                used_fix = True
                if i + 1 < len(graphemes) and graphemes[i+1] != self.morpheme_delimiter:
                    result.append(self.morpheme_delimiter)
            else:
                result.append(graphemes[i])

        if used_fix:
            print(f"Bugfix in word:\t{''.join(graphemes)}\t{''.join(result)}")
            return result
        else:
            return graphemes

    def morphemes(self, word: str) -> List[str]:
        graphemes = self.handle_analyzer_bug(self.graphemes(word))
        return super().morphemes("".join(graphemes))

    def graphemes(self, morpheme: str) -> List[str]:
        yupik_graphemes = YupikMorphemeTokenizer.to_list(morpheme)

        caps: str = YupikMorphemeTokenizer.caps_status(morpheme)

        doubled_fricative = ['ll', 'rr', 'gg', 'ghh', 'ghh*', 'ghhw']
        doubleable_fricative = ['l', 'r', 'g', 'gh', 'gh*', 'ghw']
        doubleable_nasal = ['n', 'm', 'ng', 'ngw']
        undoubleable_unvoiced_consonant = ['f', 'p', 's', 't', 'k', 'kw', 'q', 'qw']

        i = 0
        j = 1

        while j < len(yupik_graphemes):

            first = yupik_graphemes[i]
            second = yupik_graphemes[j]

            if first == self.morpheme_delimiter:
                i = j
                j = j + 1

            elif second == self.morpheme_delimiter:
                j = j + 1

            # Rule 1a
            elif first.lower() in doubleable_fricative and second.lower() in undoubleable_unvoiced_consonant:
                yupik_graphemes[i] = YupikMorphemeTokenizer.double(first, caps, i)
                i = j + 1
                j = i + 1

            # Rule 1b
            elif first.lower() in undoubleable_unvoiced_consonant and second.lower() in doubleable_fricative:
                yupik_graphemes[j] = YupikMorphemeTokenizer.double(second, caps, j)
                i = j + 1
                j = i + 1

            # Rule 2
            elif first.lower() in undoubleable_unvoiced_consonant and second.lower() in doubleable_nasal:
                yupik_graphemes[j] = YupikMorphemeTokenizer.double(second, caps, j)
                i = j + 1
                j = i + 1

            # Rule 3a
            elif first.lower() in doubled_fricative and (second.lower() in doubleable_fricative or
                                                         second.lower() in doubleable_nasal):
                yupik_graphemes[j] = YupikMorphemeTokenizer.double(second, caps, j)
                i = j + 1
                j = i + 1

            # Rule 3b
            elif ((first.lower() in doubleable_fricative or first.lower() in doubleable_nasal) and
                  second.lower() == 'll'):
                yupik_graphemes[i] = YupikMorphemeTokenizer.double(first, caps, i)
                i = j + 1
                j = i + 1

            else:
                i = j
                j = j + 1

        return yupik_graphemes


def configure(arguments: List[str]) -> argparse.Namespace:

    p = configargparse.get_argument_parser(description="Construct corpus tokenizer")

    p.add('-c', '--config', required=False, is_config_file=True, type=str, metavar='FILENAME',
          help='configuration file')

    p.add('--language', required=False, type=str, metavar='LANG',
          help='ISO 639-3 Languague Code')

    p.add('--morpheme_delimiter', required=True, type=str, metavar='STRING',
          help="In the user-provided input file, this character must appear between adjacent morphemes. " +
               "This symbol should not appear in the alphabet")

    p.add('--use_nltk_tokenizer', required=False, type=str, metavar='BOOL', default=True,
          help="Specifies whether the NLTK tokenizer be used to tokenize sentences into words.")

    p.add('-o', '--output_file', required=True, type=str, metavar="FILENAME",
          help="Output file where pickled Tokenizer object will be saved")

    return p.parse_args(args=arguments)


def main(args: argparse.Namespace):

    import pickle

    if args.language == "ess":
        tokenizer = YupikMorphemeTokenizer(morpheme_delimiter=args.morpheme_delimiter,
                                           use_nltk_tokenizer=args.use_nltk_tokenizer)
    else:
        tokenizer = MorphemeTokenizer(morpheme_delimiter=args.morpheme_delimiter,
                                      use_nltk_tokenizer=args.use_nltk_tokenizer)

    with open(args.output_file, 'wb') as output_file:

        pickle.dump(tokenizer, output_file)


if __name__ == "__main__":

    import sys

    main(configure(arguments=sys.argv[1:]))
