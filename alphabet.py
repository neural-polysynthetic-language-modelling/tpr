import logging
from typing import AbstractSet, Iterable, List, Mapping, MutableMapping, Set, Union
import unicodedata
import pdb


class Symbol:

    def __init__(self, *,
                 id: int,
                 string: str,
                 features: AbstractSet[str],
                 alphabet: 'Alphabet'):

        self._i = id
        self._s = string
        self._f = features
        self._a = alphabet

    @property
    def id(self) -> int:
        return self._i

    @property
    def string(self) -> str:
        return self._s

    @property
    def features(self) -> AbstractSet[str]:
        return self._f

    @property
    def char_vector(self) -> List[int]: # one-hot
        return self._a.char_vector(self)

    @property
    def feature_vector(self) -> List[int]: # multi-hot
        return self._a.feature_vector(self)

    @property
    def vector(self) -> List[int]:
        return self.char_vector + self.feature_vector

    def __str__(self) -> str:
        return self.string

    def __repr__(self) -> str:
        return f"Symbol(" + \
               f"id={str(self.id).zfill(len(str(len(self._a))))}, " + \
               f"char_vector={str(self.char_vector)}, " + \
               f"feature_vector={str(self.feature_vector)}, " + \
               f"string={self.string})"


class Alphabet:

    def __init__(self, *,
                 pad: str,
                 oov: str,
                 start_of_sequence: str,
                 end_of_sequence: str,
                 source: Iterable[str]) -> None:

        symbols: Mapping[str, AbstractSet[str]] = Alphabet._read_symbols(source)
        self._features: Mapping[str, int] = Alphabet._gather_features(symbols)

        errors = Alphabet.validate_parameters(pad, oov, start_of_sequence, end_of_sequence, symbols.keys())
        if len(errors) > 0:
            raise ValueError("\n".join(errors))

        self._pad = Symbol(id=0, string=pad, features=set(), alphabet=self)
        self._oov = Symbol(id=1, string=oov, features=set(), alphabet=self)
        self._start = Symbol(id=2, string=start_of_sequence, features=set(), alphabet=self)
        self._end = Symbol(id=3, string=end_of_sequence, features=set(), alphabet=self)

        self._char2id = {pad: 0, oov: 1, start_of_sequence: 2, end_of_sequence: 3}
        self._id2char = {0: pad, 1: oov, 2: start_of_sequence, 3: end_of_sequence}
        self._id2symbol = {0: self._pad, 1: self._oov, 2: self._start, 3: self._end}

        for string in symbols.keys():
            if string not in self._char2id.keys():
                id = len(self._id2symbol)

                self._char2id[string] = id
                self._id2char[id] = string
                self._id2symbol[id] = Symbol(id=id, string=string, features=symbols[string], alphabet=self)

    @staticmethod
    def load(filename: str) -> 'Alphabet':
        import pickle
        with open(filename, 'rb') as pickled_file:
            return pickle.load(pickled_file)

    def dump(self, filename: str) -> None:
        import pickle
        with open(filename, 'wb') as pickled_file:
            pickle.dump(self, pickled_file)

    def char_vector(self, symbol: Symbol) -> List[int]:
        one_hot: List[int] = [0] * len(self)
        one_hot[symbol.id] = 1
        return one_hot

    def feature_vector(self, symbol: Symbol) -> List[int]:
        multi_hot: List[int] = [0] * len(self._features)
        for feature in symbol.features:
            feature_index = self._features[feature]
            multi_hot[feature_index] = 1
        return multi_hot

    @staticmethod
    def _read_symbols(source: Iterable[str]) -> Mapping[str, AbstractSet[str]]:

        #pdb.set_trace()
        symbols: MutableMapping[str, AbstractSet[str]] = dict()

        for line_number, line in enumerate(source):
            ok = True
            parts: List[str] = line.strip().split()
            if parts[0][0] == '<' or line_number == 0:
                continue
            for part in parts:
                for character in part:  # type: str
                    category: str = unicodedata.category(character)
                    if category[0] == "Z":
                        ok = False
                        logging.warning(f"WARNING - Skipping line {line_number}; contains whitespace character:" +
                                        f"\t{Alphabet.unicode_info(character)}")
                    elif category[0] == "C":
                        ok = False
                        logging.warning(f"WARNING - Skipping line {line_number}; contains control character:" +
                                        f"\t{Alphabet.unicode_info(character)}")
            #pdb.set_trace()
            if ok:
                symbol = parts[0]
                features = set(parts[1:])
                symbols[symbol] = features

        return symbols

    @staticmethod
    def _gather_features(symbols: Mapping[str, AbstractSet[str]]) -> Mapping[str, int]:
        feature_set: Set[str] = set()
        for subset in symbols.values():
            feature_set.update(subset)

        feature_map: MutableMapping[str, int] = dict()
        for feature in sorted(feature_set):
            feature_map[feature] = len(feature_map)

        return feature_map

    @property
    def oov(self) -> Symbol:
        return self._oov

    @property
    def pad(self) -> Symbol:
        return self._pad

    @property
    def start_of_sequence(self) -> Symbol:
        return self._start

    @property
    def end_of_sequence(self) -> Symbol:
        return self._end

    def __len__(self):
        return len(self._id2symbol)

    def __getitem__(self, key: Union[int, str]) -> Symbol:

        if isinstance(key, int):
            if self.__contains__(key):
                return self._id2symbol[key]
            return self.oov

        if isinstance(key, str):
            if self.__contains__(key):
                return self._id2symbol[self._char2id[key]]
            return self.oov

        raise TypeError(f"Alphabet key must be int or str, not {type(key)}")

    def __iter__(self):
        return iter (list(self._id2symbol.values ()))

    def __contains__(self, item):

        if isinstance(item, int):
            return item in self._id2symbol
        elif isinstance(item, str):
            return item in self._char2id
        elif isinstance(item, Symbol) :
            return item in list (self._id2symbol.values ())
        raise TypeError(f"Item must be int or str or Symbol, not {type(item)}")

    @property
    def characters (self) -> AbstractSet[str]:
        return set (map (lambda x: x.string, list (self._id2symbol .values ())))

    @staticmethod
    def char_to_code_point(c: str) -> str:
        x_hex_string: str = hex(ord(c))  # a string of the form "0x95" or "0x2025"
        hex_string: str = x_hex_string[2:]  # a string of the form "95" or "2025"
        required_zero_padding = max(0, 4 - len(hex_string))
        return (
            f"U+{required_zero_padding * '0'}{hex_string}"
        )  # a string of the form "\\u0095" or "\\u2025"

    @staticmethod
    def char_to_name(c: str) -> str:
        try:
            return unicodedata.name(c)
        except ValueError:
            return ""

    @staticmethod
    def unicode_info(s: str) -> str:
        return (
            s
            + "\t"
            + "; ".join(
                [
                    f"{Alphabet.char_to_code_point(c)} {Alphabet.char_to_name(c)}"
                    for c in s
                ]
            )
        )

    @staticmethod
    def validate_parameters(pad: str,
                            oov: str,
                            start_of_sequence: str,
                            end_of_sequence: str,
                            symbols: AbstractSet[str]):

        errors: List[str] = list()

        if pad in symbols:
            errors.append(f"The padding symbol ({Alphabet.unicode_info(pad)})" +
                          f" must not be in the provided set of symbols.")

        if oov in symbols:
            errors.append(f"The padding symbol ({Alphabet.unicode_info(pad)})" +
                          f" must not be in the provided set of symbols.")

        if start_of_sequence in symbols:
            errors.append(f"The start-of-sequence symbol ({Alphabet.unicode_info(start_of_sequence)})" +
                          f" must not be in the provided set of symbols")

        if end_of_sequence in symbols:
            errors.append(f"The end-of-sequence symbol ({Alphabet.unicode_info(end_of_sequence)})" +
                          f" must not be in the provided set of symbols")

        if pad == oov:
            errors.append(f"The padding symbol ({Alphabet.unicode_info(pad)})" +
                          f" and the out-of-vocabulary symbol ({Alphabet.unicode_info(oov)})" +
                          f" must not be the same.")

        if pad == start_of_sequence:
            errors.append(f"The padding symbol ({Alphabet.unicode_info(pad)})" +
                          f" and the start-of-sequence symbol ({Alphabet.unicode_info(start_of_sequence)})" +
                          f" must not be the same.")

        if pad == end_of_sequence:
            errors.append(f"The padding symbol ({Alphabet.unicode_info(pad)})" +
                          f" and the end-of-sequence symbol ({Alphabet.unicode_info(end_of_sequence)})" +
                          f" must not be the same.")

        if oov == start_of_sequence:
            errors.append(f"The out-of-vocabulary symbol ({Alphabet.unicode_info(oov)})" +
                          f" and the start-of-sequence symbol ({Alphabet.unicode_info(start_of_sequence)})" +
                          f" must not be the same.")

        if oov == end_of_sequence:
            errors.append(f"The out-of-vocabulary symbol ({Alphabet.unicode_info(oov)})" +
                          f" and the end-of-sequence symbol ({Alphabet.unicode_info(end_of_sequence)})" +
                          f" must not be the same.")

        if start_of_sequence == end_of_sequence:
            errors.append(f"The start-of-sequence symbol ({Alphabet.unicode_info(start_of_sequence)})" +
                          f" and the end-of-sequence symbol ({Alphabet.unicode_info(end_of_sequence)})" +
                          f" must not be the same.")

        return errors


if __name__ == '__main__':

    import pickle
    import sys

    if len(sys.argv) != 3:
        print(f"Usage:\t{sys.argv[0]} one_symbol_per_line.txt vocab.pkl")
        sys.exit(-1)

    with open(sys.argv[1]) as symbol_file, open(sys.argv[2], 'wb') as pickle_file:

        print(f"Constructing alphabet from {sys.argv[1]}...", file=sys.stderr)
        vocab = Alphabet(pad='<pad>',
                         oov='<oov>',
                         start_of_sequence='<s>',
                         end_of_sequence='</s>',
                         source=symbol_file)

        print(f"Writing alphabet of {len(vocab)} symbols to {sys.argv[2]}...", file=sys.stderr)
        pickle.dump(vocab, pickle_file)

        print(f"Completed writing alphabet of {len(vocab)} symbols to {sys.argv[2]}", file=sys.stderr)
