from typing import List, MutableMapping, NamedTuple

from features import Alphabet, Symbol


class Morpheme(NamedTuple):
    graphemes: List[Symbol]
    tpr: List[List[int]]

    def __len__(self) -> int:
        return len(self.graphemes)

    def __str__(self) -> str:
        return "".join(self.graphemes)


class Morphemes:

    def __init__(self, *,
                 alphabet: Alphabet, max_len: int,
                 start_of_morpheme: str, end_of_morpheme: str):
        if start_of_morpheme not in alphabet or end_of_morpheme not in alphabet:
            raise ValueError(f"Start-of-morpheme symbol {start_of_morpheme}" +
                             f" and end-of-morpheme symbol {end_of_morpheme} must both be present in the alphabet")
        self.alphabet: Alphabet = alphabet
        self.max_len: int = max_len
        self.start_of_morpheme: Symbol = alphabet[start_of_morpheme]
        self.end_of_morpheme: Symbol = alphabet[end_of_morpheme]
        self._grapheme_delimiter: str = "\u001F"
        self._dictionary: MutableMapping[str, Morpheme] = dict()

    def _get_key(self, graphemes: List[str]) -> str:
        if isinstance(graphemes, list) and self._grapheme_delimiter not in graphemes:
            return self._grapheme_delimiter.join(graphemes)
        else:
            raise ValueError

    def _get_tpr(self, graphemes: List[Symbol]) -> List[List[int]]:
        if len(graphemes) > self.max_len:
            raise ValueError(f"Morpheme {str(graphemes)} has length longer than maximum allowed {self.max_len}")

        num_pads = self.max_len - len(graphemes)

        symbols: List[Symbol] = ([self.start_of_sequence] +
                                 graphemes +
                                 [self.end_of_sequence] +
                                 [self.alphabet.pad] * num_pads)

        return [symbol.vector for symbol in symbols]

    def add(self, graphemes: List[str]) -> Morpheme:
        key: str = self._get_key(graphemes)
        symbols: List[Symbol] = [self.alphabet[grapheme] for grapheme in graphemes]
        tpr: List[List[int]] = self._get_tpr(symbols)
        morpheme = Morpheme(symbols, tpr)
        self._dictionary[key] = morpheme
        return morpheme

    def __getitem__(self, graphemes: List[str]) -> Morpheme:
        key: str = self._get_key(graphemes)
        return self._dictionary[key]

    def __len__(self) -> int:
        return len(self._dictionary)

    def __contains__(self, graphemes: List[str]) -> bool:
        key: str = self._get_key(graphemes)
        return key in self._dictionary

