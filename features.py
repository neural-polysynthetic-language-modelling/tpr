from itertools import accumulate
from typing import Iterator, List, Mapping, Optional, Set


class Feature:
    """
    Encapsulates a feature and the set of values that feature may take,
    including the value used to represent the absence of that feature.
    """

    def __init__(self, *, name: str, values: Set[str], oov: Optional[str] = None, start_at: int = 0):
        """
        Constructs an object encapsulating a feature and its associated string and integer values.

        An example of a feature follows:
        Feature(name='Place of Articulation',
                values={'', 'labial', 'alveolar', 'velar', 'uvular})

        This constructor will assign integer values to each of the provided values,
        starting with the value provided by the start_at parameter, and increasing sequentially.
        """
        if oov is not None and oov not in values:
            raise ValueError(f"Parameter oov='{oov}' must be present in values but is not.")
        self.name: str = name
        self.oov: Optional[str] = oov
        self.num_values = len(values)
        self.features = sorted(values) if oov is None else [oov] + sorted(values.difference([oov]))
        self.int2value: Mapping[int, str] = {integer: feature_value
                                             for (integer, feature_value) in enumerate(iterable=self.features,
                                                                                       start=start_at)}
        self.value2int: Mapping[str, int] = {feature_value: integer
                                             for (integer, feature_value) in self.int2value.items()}

    def __len__(self) -> int:
        return self.num_values

    def __iter__(self):
        return iter(self.features)

    def __getitem__(self, key):
        if isinstance(key, int):
            if key in self.int2value:
                return self.int2value[key]
            else:
                raise ValueError(f"Key {key} is not valid for Feature({self.name})")
        elif isinstance(key, str):
            if key in self.value2int:
                return self.value2int[key]
            elif self.oov is not None:
                return self.value2int[self.oov]
            else:
                raise ValueError(f"Key {key} is not valid for Feature({self.name})")
        else:
            raise ValueError

    def __str__(self) -> str:
        return f"Feature({self.name}, size={len(self)}, start={self[self.features[0]]}, end={self[self.features[-1]]})"


class Features:

    def __init__(self, features: List[Feature]):
        self.features = features

    def __iter__(self):
        return iter(self.features)

    def __getitem__(self, index: int) -> Feature:
        return self.features[index]

    def __len__(self) -> int:
        return sum([len(feature) for feature in self.features])


class Symbol:

    def __init__(self, *, features: Features, values: List[str]):
        if len(values) != len(features.features):
            raise ValueError(f"Number of values must match number of features:" +
                             f" {len(values)} != {len(features.features)}")
        self.features = features
        self.values = values
        self.vector: List[int] = [0] * len(self.features)
        for index in [self.features[i][self.values[i]] for i in range(len(self.values))]:
            self.vector[index] = 1

    def __str__(self) -> str:
        return self.values[0]

    def __repr__(self) -> str:
        return f"Symbol(" + \
               f"vector={str(self.vector)}, " + \
               f"values={', '.join(self.values)})"


class Alphabet:

    def __init__(self, *, symbols: List[Symbol], oov: Symbol, pad: Symbol):
        self.symbols = symbols
        self.mapping = {str(symbol): symbol for symbol in symbols}
        self.oov = oov
        self.pad = pad

    def __iter__(self):
        return iter(self.symbols)

    def __len__(self) -> int:
        return len(self.symbols)

    def __getitem__(self, key: str) -> Symbol:
        if key in self.mapping:
            return self.mapping[key]
        else:
            return self.oov

    @staticmethod
    def load(filename: str) -> 'Alphabet':
        import pickle
        with open(filename, 'rb') as pickled_file:
            return pickle.load(pickled_file)

    def dump(self, filename: str) -> None:
        import pickle
        with open(filename, 'wb') as pickled_file:
            pickle.dump(self, pickled_file)


def construct_yupik_alphabet(source: Iterator[str]) -> Alphabet:

    # The source should consist of tab-separated values, and the first line should be feature names
    feature_names: List[str] = [name.strip() for name in next(source).split('\t')]

    # Read the rest of the lines such that raw_feature_values[l][c] contains the feature value for line l, column c
    raw_feature_values: List[List[str]] = [[value.strip() for value in line.split('\t')] for line in source]

    # Transpose, unique-ify, and sort the feature values such that
    #   unique_feature_values[f] contains set of feature values for the f^th feature.
    unique_feature_values: List[Set[str]] = [{raw_feature_values[line][column]
                                              for line in range(len(raw_feature_values))}
                                             for column in range(len(feature_names))]

    # For each feature, keep track of how many feature values have been cumulatively seen before we reach that feature
    offsets: List[int] = ([0] +
                          list(accumulate([len(unique_feature_values[f]) for f in range(len(feature_names))]))
                          )[:-1]

    # The first feature is taken to be the OOV symbol.
    # For all other features the empty string represents the absence of that feature.
    oovs: List[Optional[str]] = [raw_feature_values[0][0]] + [None for _ in range(len(feature_names)-1)]

    # Create list of Feature objects
    features: Features = Features([Feature(name=feature_names[f],
                                           values=unique_feature_values[f],
                                           start_at=offsets[f],
                                           oov=oovs[f])
                                   for f in range(len(feature_names))])

    symbols: List[Symbol] = [Symbol(features=features, values=values) for values in raw_feature_values]

    alphabet: Alphabet = Alphabet(symbols=symbols, oov=symbols[0], pad=symbols[1])

    return alphabet


def main(tsv_filename: str, pickle_filename: str) -> None:

    import sys
    import pickle

    with open(tsv_filename, 'rt') as tsv_file, open(pickle_filename, 'wb') as pickle_file:
        print(f"Constructing alphabet from {tsv_filename}...", file=sys.stderr)
        alphabet: Alphabet = construct_yupik_alphabet(source=tsv_file)

        print(f"Writing alphabet of {len(alphabet)} symbols to {pickle_filename}...", file=sys.stderr)
        pickle.dump(alphabet, pickle_file)

        print(f"Completed writing alphabet of {len(alphabet)} symbols to {pickle_filename}", file=sys.stderr)


if __name__ == '__main__':

    import sys

    if len(sys.argv) != 3:
        print(f"Usage:\t{sys.argv[0]} vocab_with_features.tsv vocab.pkl")
        sys.exit(-1)

    main(tsv_filename=sys.argv[1], pickle_filename=sys.argv[2])
