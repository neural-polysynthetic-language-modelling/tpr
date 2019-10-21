from typing import Iterable, MutableMapping, NamedTuple, Union


class Symbol(NamedTuple):
    integer: int
    string: str


class Alphabet:

    def __init__(self, *,
                 pad: str,
                 oov: str,
                 start_of_sequence: str,
                 end_of_sequence: str,
                 strings: Iterable[str]) -> None:

        self._pad = Symbol(integer=0, string=pad)
        self._oov = Symbol(integer=1, string=oov)
        self._start = Symbol(integer=2, string=start_of_sequence)
        self._end = Symbol(integer=3, string=end_of_sequence)

        self._entry_map: MutableMapping[Union[int, str], Symbol] = {0: self._pad, pad: self._pad,
                                                                    1: self._oov, oov: self._oov,
                                                                    2: self._start, start_of_sequence: self._start,
                                                                    3: self._end, end_of_sequence: self._end}

        self._entry_list = [self._pad, self._oov, self._start, self._end]

        for string in strings:

            if string == pad or string == oov or string == start_of_sequence or string == end_of_sequence:
                raise ValueError(f"Strings must not contain reserved string:\t{string}")

            if string not in self._entry_map:
                v = Symbol(integer=len(self._entry_list), string=string)
                self._entry_map[v.integer] = v
                self._entry_map[v.string] = v
                self._entry_list.append(v)

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


if __name__ == '__main__':

    import pickle
    import sys

    if len(sys.argv) != 3:
        print(f"Usage:\t{sys.argv[0]} one_symbol_per_line.txt vocab.pkl")
        sys.exit(-1)

    with open(sys.argv[1]) as symbol_file, open(sys.argv[2], 'wb') as pickle_file:

        print(f"Reading symbols from {sys.argv[1]}...", file=sys.stderr)
        symbols = [line.strip() for line in symbol_file]

        print(f"Constructing vocabulary from symbols...", file=sys.stderr)
        vocab = Alphabet(pad='<pad>',
                         oov='<oov>',
                         start_of_sequence='<s>',
                         end_of_sequence='</s>',
                         strings=symbols)

        print(f"Writing vocabulary of {len(vocab)} symbols to {sys.argv[2]}...", file=sys.stderr)
        pickle.dump(vocab, pickle_file)

        print(f"Completed writing vocabulary of {len(vocab)} symbols to {sys.argv[2]}", file=sys.stderr)
