from alphabet import Alphabet, Symbol

if __name__ == '__main__':

    import pickle
    import sys

    if len(sys.argv) != 2:
        print(f"Usage:\t{sys.argv[0]} vocab.pkl")
        sys.exit(-1)

    with open(sys.argv[1], 'rb') as pickle_file:

        print(f"Reading vocabulary from {sys.argv[1]}...", file=sys.stderr)
        vocab: Alphabet = pickle.load(pickle_file)

        for symbol in vocab:  # type: Symbol
            print(repr(symbol))
