import logging
from typing import AbstractSet, Iterable, List, Mapping, MutableMapping, Set, Union
import unicodedata


class Symbol:

    def __init__(self, *,
                 integer: int,
                 string: str,
                 features: AbstractSet[str],
                 alphabet: 'Alphabet'):

        self._i = integer
        self._s = string
        self._f = features
        self._a = alphabet

    @property
    def integer(self) -> int:
        return self._i

    @property
    def string(self) -> str:
        return self._s

    @property
    def features(self) -> AbstractSet[str]:
        return self._f

    @property
    def vector(self) -> List[int]:
        return self._a.vector(self)

    def __str__(self) -> str:
        return self.string

    def __repr__(self) -> str:
        return f"Symbol(" + \
               f"integer={str(self.integer).zfill(len(str(len(self._a))))}, " + \
               f"vector={str(self.vector)}, " + \
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

        self._pad = Symbol(integer=0, string=pad, features=set(), alphabet=self)
        self._oov = Symbol(integer=1, string=oov, features=set(), alphabet=self)
        self._start = Symbol(integer=2, string=start_of_sequence, features=set(), alphabet=self)
        self._end = Symbol(integer=3, string=end_of_sequence, features=set(), alphabet=self)

        self._entry_map: MutableMapping[Union[int, str], Symbol] = {0: self._pad, pad: self._pad,
                                                                    1: self._oov, oov: self._oov,
                                                                    2: self._start, start_of_sequence: self._start,
                                                                    3: self._end, end_of_sequence: self._end}

        self._entry_list = [self._pad, self._oov, self._start, self._end]

        for string in symbols:

            if string not in self._entry_map:
                v = Symbol(integer=len(self._entry_list), string=string, features=symbols[string], alphabet=self)
                self._entry_map[v.integer] = v
                self._entry_map[v.string] = v
                self._entry_list.append(v)

    @staticmethod
    def load(filename: str) -> 'Alphabet':
        import pickle
        with open(filename, 'rb') as pickled_file:
            return pickle.load(pickled_file)

    def dump(self, filename: str) -> None:
        import pickle
        with open(filename, 'wb') as pickled_file:
            pickle.dump(self, pickled_file)

    def vector(self, symbol: Symbol) -> List[int]:
        one_hot: List[int] = [0] * len(self)
        one_hot[symbol.integer] = 1

        multi_hot: List[int] = [0] * len(self._features)
        for feature in symbol.features:
            feature_index = self._features[feature]
            multi_hot[feature_index] = 1

        return one_hot + multi_hot

    @staticmethod
    def _read_symbols(source: Iterable[str]) -> Mapping[str, AbstractSet[str]]:

        symbols: MutableMapping[str, AbstractSet[str]] = dict()

        for line_number, line in enumerate(source):
            ok = True
            parts: List[str] = line.strip().split()
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
        return len(self._entry_list)

    def __getitem__(self, key: Union[int, str]) -> Symbol:
        if isinstance(key, int) or isinstance(key, str):
            if key in self._entry_map:
                return self._entry_map[key]
            else:
                return self.oov
        else:
            raise TypeError(f"Alphabet key must be int or str, not {type(key)}")

    def __iter__(self):
        return iter(self._entry_list)

    def __contains__(self, item):
        if isinstance(item, int) or isinstance(item, str):
            return item in self._entry_map
        elif isinstance(item, Symbol):
            return item in self._entry_list
        else:
            return False

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
