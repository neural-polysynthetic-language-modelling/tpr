import argparse
import configargparse
import logging
from typing import List, NamedTuple, Tuple
import util
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.optim as optim
import time


def configure_training(args: List[str]) -> argparse.Namespace:

    p = configargparse.get_argument_parser()
    p.add('-c', '--config', required=False, is_config_file=True, help='configuration file')

    p.add('--training_data', required=True, help='Tab-separated training file')
    p.add('--val_data', required=True, help='Tab-separated validation file')
    p.add('--hidden_size', required=True, type=int)
    # p.add('--hidden_layers', required=True, type=int)
    p.add('--embedding_dim', required = True, type=int)
    #
    # p.add('-o', '--output_file', required=True, type=str, metavar="FILENAME",
    #       help="Output file where trained MorphemeSegmenter model will be saved")
    #
    # p.add('--continue_training', required=False, type=bool, help='Continue training')
    #
    # p.add('--print_every', required=True, type=int)
    p.add('--batch_size', required=True, type=int)
    p.add('--num_epochs', required=True, type=int)
    p.add('--learning_rate', required=True, type=float)

    return p.parse_args(args=args)

def configure_testing(args: List[str]) -> argparse.Namespace:
    pass


class Vocab_Lang():
    def __init__(self, is_y = False, morph_delim = '^'):
        self.word2idx = {'<pad>':0, '<unk>': 1, '<sos>': 2, '<eos>':3}
        self.idx2word = {0:'<pad>', 1: '<unk>', 2: '<sos>', '<eos>':3}
        self.vocab = set()
        self.morph_delim = morph_delim

        self.is_y = is_y

    def add_example(self, word): # example is a value in the left or right column of data i.e. ('writing', 'writ^ing') => example would be 'writing' or 'writ^ing'

        if self.is_y:
            vals = word.split(self.morph_delim)
        else:
            vals = list(word)

        self.vocab.update(vals)
        for word in vals:
            self.word2idx[word] = self.word2idx.get(word, len(self.word2idx.values()))
            self.idx2word[self.word2idx[word]] = word

    def get_tensor(self, example):
        if self.is_y:
            vals = example.split(self.morph_delim)
        else:
            vals = list(example)

        vals = ['<sos>'] + vals + ['<eos>']

        return [self.word2idx.get(word, self.word2idx['<unk>']) for word in vals]


class MyDataset(Dataset):

    def __init__(self, data, src_vocab = None, trg_vocab = None):

        # Learn source and target vocab
        assert src_vocab is None and trg_vocab is None or src_vocab is not None and trg_vocab is not None
        if src_vocab is None:
            self.src_vocab = Vocab_Lang(is_y=False)
            self.trg_vocab = Vocab_Lang(is_y=True)
            self.build_vocab(data)
        else:
            self.src_vocab = src_vocab
            self.trg_vocab = trg_vocab

        # Dataset specific variables that store the data
        self.X_words = [] # [write, reading, ...]
        self.y_words = [] # [write, read^ing, ...]
        self.X = None
        self.y = None
        self.X_length = None
        self.max_length_src = None
        self.max_length_trg = None


        self.build_matrices(data)

    def build_vocab(self, training_data):
        with open (training_data, 'r') as data:
            for line in data:
                splt = line.split('\t')
                if len(splt) != 2:
                    print ("Bad split!", splt)
                    continue
                self.src_vocab.add_example(splt[0])
                self.trg_vocab.add_example(splt[1])

    def build_matrices(self, data):

        with open (data, 'r') as dat:
            for line in dat:
                splt = line.split('\t')
                if len(splt) != 2:
                    print ("Bad split! - build_matrices", splt)
                    continue
                self.X_words.append(splt[0])
                self.y_words.append(splt[1])

        # Vectorize the input and target languages
        src_tensor = [self.src_vocab.get_tensor(example) for example in self.X_words]
        trg_tensor = [self.trg_vocab.get_tensor(example) for example in self.y_words]

        self.X_length = [len(vec) for vec in src_tensor] # length of sentences in src side before padding

        # calculate the max_length of input and output tensor for padding
        self.max_length_src, self.max_length_trg = max([len(vec) for vec in src_tensor]), max([len(vec) for vec in trg_tensor])

        def pad_sequences(x, max_len):
            padded = np.zeros((max_len), dtype=np.int64) # zeros b/c zero is padding idx
            if len(x) > max_len:
                padded[:] = x[:max_len]
            else:
                padded[:len(x)] = x
            return padded

        self.X = [pad_sequences(example, self.max_length_src) for example in src_tensor]
        self.y = [pad_sequences(example, self.max_length_trg) for example in trg_tensor]

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        x_len = self.X_length[idx]
        return x, y, x_len

    def __len__(self):
        assert len(self.X) == len(self.y)
        return len(self.X)

### sort batch function to be able to use with pad_packed_sequence
def sort_batch(X, y, lengths):
    lengths, indx = lengths.sort(dim=0, descending=True)
    X = X[indx]
    y = y[indx]
    return X.transpose(0,1), y, lengths # transpose (batch x seq) to (seq x batch)

def train(args: argparse.Namespace) -> None:

    device = util.get_device()

    logging.info(f"Training MorphemeSegmenter on {str(device)} using {args.training_data} as training data")

    # Set up train and dev dataloaders
    train_dataset = MyDataset(args.training_data)
    val_dataset = MyDataset(args.val_data, train_dataset.src_vocab, train_dataset.trg_vocab)
    train_dataloader = DataLoader(train_dataset, batch_size = args.batch_size, drop_last = True, shuffle = True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, drop_last=True, shuffle=False)

    # Set up model (encoder & decoder)
    vocab_src_size = len(train_dataset.src_vocab.word2idx)
    vocab_trg_size = len(train_dataset.trg_vocab.word2idx)
    encoder = Encoder(vocab_src_size, args.embedding_dim, args.hidden_size, args.batch_size)
    decoder = Decoder(vocab_trg_size, args.embedding_dim, args.hidden_size, args.batch_size)
    encoder.to(device)
    decoder.to(device)

    # Set up optimizer
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=args.learning_rate)

    # Begin training - loop num_epochs time
    for epoch in range(args.num_epochs):
        start = time.time()
        encoder.train()
        decoder.train()

        total_loss = 0

        # Loop over each batch (at the end of each batch, update parameters i.e. backpropagate)
        for (batch, (inp, targ, inp_len)) in enumerate(train_dataloader):
            loss = 0

            # Prepare inputs and run encoder
            inp, targ, inp_len = inp, targ, inp_len # batch_size x max_length_src; batch_size x max_length_trg; batch_size
            xs, ys, lens = sort_batch(inp, targ, inp_len)  # max_length_src x batch_size; batch_size x max_length_trg; batch_size
            enc_output, enc_hidden = encoder(xs.to(device), lens, device = device) # max_length_src x batch_size x hidden_size; batch_size x hidden_size

            # Initialize inputs to decoder
            dec_hidden = enc_hidden # batch_size x hidden_size
            dec_input = torch.tensor([[train_dataset.trg_vocab.word2idx['<sos>']]] * args.batch_size)  # batch_size x 1

            # Run decoder one step at a time, using teacher forcing
            for t in range(1, ys.size(1)):
                predictions, dec_hidden, _ = decoder(dec_input.to(device),  # batch_size x 1
                                                     dec_hidden.to(device),  # batch_size x hidden_size
                                                     enc_output.to(device))  # MAX_LEN_SRC x batch_size x hidden_size
                # batch_size x vocab_size; batch_size x 1024; batch_size x 1 x MAX_LEN_SRC






    # dataset = DataLoader(MyDataset(), batch_size=BATCH_SIZE,
    #                      drop_last=True,
    #                      shuffle=True)

    # val_dataset = DataLoader(val_dataset, batch_size=BATCH_SIZE,
    #                          drop_last=True,
    #                          shuffle=False)

    #dataloader = MyDataLoader(args.training_data)

    # model: MorphemeVectors = MorphemeVectors(
    #     corpus=MorphemeCorpus.load(args.corpus),
    #     hidden_layer_size=args.hidden_size,
    #     num_hidden_layers=args.hidden_layers,
    #     device=device)
    #
    # model.run_training(learning_rate=args.learning_rate,
    #                    epochs=args.num_epochs,
    #                    batch_size=args.batch_size,
    #                    logging_frequency=args.print_every)
    #
    # logging.info(f"Saving model to {args.output_file}")
    # model.to(torch.device("cpu"))
    # torch.save(model, args.output_file)




if __name__ == "__main__":

    import sys
    #
    # logging.basicConfig(
    #     level='INFO',
    #     stream=sys.stderr,
    #     datefmt="%Y-%m-%d %H:%M:%S",
    #     format="%(asctime)s\t%(message)s",
    # )
    #
    # if '--num_epochs' in sys.argv:
    #     train(configure_training(args=sys.argv[1:]))
    # else:
    #     evaluate(configure_testing(args=sys.argv[1:]))

    args = configure_training(sys.argv[1:])
    dataset = MyDataset(args.training_data)
    dev_dataset = MyDataset(args.val_data, dataset.src_vocab, dataset.trg_vocab)

    print (dataset.src_vocab.word2idx)
    print (dataset.y_words[0:2])
    print (dataset.y[0:2])
    print ('-------')
    print (dev_dataset.src_vocab.word2idx)
    print (dev_dataset.y_words[0:2])
    print (dev_dataset.y[0:2])