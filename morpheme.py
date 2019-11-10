import itertools
from typing import Any, List, Mapping, NamedTuple

from features import Alphabet, Symbol


class Morpheme(NamedTuple):
    graphemes: List[Symbol]
    tpr: List[List[int]]

    def __len__(self) -> int:
        return len(self.graphemes)

    def __str__(self) -> str:
        return Morphemes.list_to_string(self.graphemes, delimiter='')

    @property
    def flattened_tpr(self) -> List[int]:
        return list(itertools.chain.from_iterable(self.tpr))

    @property
    def shape(self) -> List[int]:
        return list(len(self.tpr), len(self.tpr[0]))


class Morphemes:

    def __init__(self, *,
                 alphabet: Alphabet,
                 start_of_morpheme: str, end_of_morpheme: str,
                 list_of_morphemes: List[List[str]]):
        if start_of_morpheme not in alphabet or end_of_morpheme not in alphabet:
            raise ValueError(f"Start-of-morpheme symbol {start_of_morpheme}" +
                             f" and end-of-morpheme symbol {end_of_morpheme} must both be present in the alphabet")
        self.alphabet: Alphabet = alphabet
        self.max_len: int = len(max(list_of_morphemes, key=len))
        self.start_of_morpheme: Symbol = alphabet[start_of_morpheme]
        self.end_of_morpheme: Symbol = alphabet[end_of_morpheme]
        self._grapheme_delimiter: str = "\u001F"

        morpheme_symbols: List[List[Symbol]] = [[alphabet[grapheme] for grapheme in graphemes]
                                                for graphemes in list_of_morphemes]

        tprs: List[List[List[int]]] = [Morphemes.tpr(morpheme, self.max_len,
                                                     self.start_of_morpheme, self.end_of_morpheme, alphabet.pad)
                                       for morpheme in morpheme_symbols]

        self.morpheme_list: List[Morpheme] = [Morpheme(graphemes, tpr)
                                              for (graphemes, tpr) in zip(morpheme_symbols, tprs)]

        self.morpheme_map: Mapping[str, Morpheme] = {Morphemes.list_to_string(morpheme.graphemes,
                                                                              self._grapheme_delimiter): morpheme
                                                     for morpheme in self.morpheme_list}

    @property
    def flattened_tpr_size(self) -> int:
        return len(self.morpheme_list[0].flattened_tpr)

    @property
    def tpr_shape(self) -> List[int]:
        return self.morpheme_list[0].shape

    @staticmethod
    def tpr(graphemes: List[Symbol], max_len: int,
            start_of_sequence: Symbol, end_of_sequence: Symbol, pad: Symbol) -> List[List[int]]:
        if len(graphemes) > max_len:
            raise ValueError(f"Morpheme {str(graphemes)} has length longer than maximum allowed {max_len}")

        num_pads = max_len - len(graphemes)

        symbols: List[Symbol] = ([start_of_sequence] +
                                 graphemes +
                                 [end_of_sequence] +
                                 [pad] * num_pads)

        return [symbol.vector for symbol in symbols]

    @staticmethod
    def list_to_string(graphemes: List[Any], delimiter: str) -> str:
        if isinstance(graphemes, list) and delimiter not in graphemes:
            return delimiter.join([str(grapheme) for grapheme in graphemes])
        else:
            raise ValueError

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.morpheme_list[key]
        elif isinstance(key, list):
            key: str = Morphemes.list_to_string(key, self._grapheme_delimiter)
            return self.morpheme_map[key]
        else:
            return ValueError

    def __len__(self) -> int:
        return len(self.morpheme_list)

    def __contains__(self, graphemes: List[str]) -> bool:
        key: str = Morphemes.list_to_string(graphemes, self._grapheme_delimiter)
        return key in self.morpheme_map
