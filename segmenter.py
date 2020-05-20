import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import unicodedata
import re
import pdb

import time
import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction

import configargparse
from torch.utils.data import Dataset, DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# MAX_TRG_LENGTH = 11
MAX_TRG_LENGTH = 55

def configure_training(args): # -> argparse.Namespace: #: List[str]) -> argparse.Namespace:

    p = configargparse.get_argument_parser()
    p.add('-c', '--config', required=False, is_config_file=True, help='configuration file')

    p.add('--data', required=True, help='Tab-separated training file')
    # p.add('--dev_data', required=False, help='Tab-separated validation file')
    # p.add('--test_data', required=False, help='Tab-separated test file')

    p.add('--lang', required=True, help='esp or ess')
    p.add('--num_epochs', required=True, type=int)

    return p.parse_args(args=args)
## Feel free to change any parameters class definitions as long as you can change the training code, but make sure
## evaluation should get the tensor format it expects

class Model(object):
    def __init__(self, vocab_inp_size, vocab_trg_size, embedding_dim, units, batch_sz):
        self.encoder = Encoder(vocab_inp_size, embedding_dim, units, batch_sz)
        self.decoder = Decoder(vocab_trg_size, embedding_dim, units, batch_sz)
        self.encoder.to(DEVICE)
        self.decoder.to(DEVICE)

        self.vocab_inp_size = vocab_inp_size
        self.vocab_trg_size = vocab_trg_size
        self.embedding_dim = embedding_dim
        self.units = units
        self.batch_sz = batch_sz

class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, batch_sz):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(embedding_dim, hidden_size) #, bidirectional=True)

    def forward(self, x, lens):
        '''
        Pseudo-code
        - Pass x through an embedding layer
        - Make sure x is correctly packed before the recurrent net
        - Pass it through the recurrent net
        - Make sure the output is unpacked correctly
        - return output and hidden states from the recurrent net
        - Feel free to play around with dimensions - the training loop should help you determine the dimensions
        '''

        x = x  # MAX_LEN_SRC x batch_size
        embeds = self.embedding(x)  # MAX_LEN_SRC x batch_size x embedding_dim
        packed_input = pack_padded_sequence(embeds, lens)  # (sum(lens) x embedding_dim; max(lens))
        output, hidden = self.gru(
            packed_input)  # (sum(lens) x hidden_size; max(lens)); num_layers x batch_size x hidden_size
        output, xxx = pad_packed_sequence(output, total_length=x.shape[
            0])  # MAX_LEN_SRC x batch_size x hidden_size; batch_size (same as lens)
        hidden = hidden[0]  # batch_size x hidden_size

        return output, hidden  # MAX_LEN_SRC x batch_size x hidden_size; batch_size x hidden_size


## Feel free to change any parameters class definitions as long as you can change the training code, but make sure
## evaluation should get the tensor format it expects
class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, batch_sz):
        super(Decoder, self).__init__()
        self.attn = nn.Linear(hidden_size, hidden_size)
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.emb_dropout = nn.Dropout(0.1)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.attn_combine = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden, enc_output): #pred end is a tensor
        '''
        Pseudo-code
        - Calculate the score using the formula shown above using encoder output and hidden output.
        Note h_t is the hidden output of the decoder and h_s is the encoder output in the formula
        - Calculate the attention weights using softmax and
        passing through V - which can be implemented as a fully connected layer
        - Finally find c_t which is a context vector where the shape of context_vector should be (batch_size, hidden_size)
        - You need to unsqueeze the context_vector for concatenating with x aas listed in Point 3 above
        - Pass this concatenated tensor to the GRU and follow as specified in Point 4 above

        Returns :
        output - shape = (batch_size, vocab)
        hidden state - shape = (batch_size, hidden size)
        '''
        x, hidden, enc_output = x, hidden, enc_output  # batch_size x 1; # batch_size x hidden_size; # MAX_LEN_SRC x batch_size x hidden_size

        embedded = self.emb_dropout(self.embedding(x))  # batch_size x 1 x hidden_size
        embedded = embedded.transpose(1, 0)  # 1 x batch_size x hidden_size
        hidden = hidden.unsqueeze(0)  # 1 x batch_size x hidden_size
        rnn_output, new_hidden = self.gru(embedded, hidden)  # 1 x batch_size x 1024; 1 x batch_size x 1024

        energy = self.attn(enc_output)  # MAX_LEN_SRC x batch_size x hidden_size      corresponds to Wh_s
        attn_energies = torch.sum(rnn_output * energy, dim=2)  # MAX_LEN_SRC x batch_size     corresponds to h_t^t W h_S
        attn_energies = attn_energies.t()  # batch_size x MAX_LEN_SRC
        attn_weights = F.softmax(attn_energies, dim=1).unsqueeze(1)  # batch_size x 1 x MAX_LEN_SRC

        enc_output_t = enc_output.transpose(0, 1)  # batch_size x MAX_LEN_SRC x hidden_size
        context = attn_weights.bmm(enc_output_t)  # batch_size x 1 x hidden_size

        rnn_output = rnn_output.squeeze(0)  # batch_size x hidden_size
        context = context.squeeze(1)  # batch_size x hidden_size
        concat_input = torch.cat((rnn_output, context), 1)  # batch_size x 2*hidden_size
        concat_output = torch.tanh(self.attn_combine(concat_input))  # batch_size x hidden_size

        output = self.out(concat_output)  # batch_size x vocab_size
        output = F.softmax(output, dim=1)  # batch_size x vocab_size

        return output, new_hidden.squeeze(
            0), attn_weights  # batch_size x vocab_size; batch_size x 1024; batch_size x 1 x MAX_LEN_SRC

def loss_function(real, pred): #, criterion):
    """ Only consider non-zero inputs in the loss; mask needed """
    # mask = 1 - np.equal(real, 0) # assign 0 to all above 0 and 1 to all 0s
    # print(mask)
    mask = real.ge(1).type(torch.cuda.FloatTensor)
    criterion = nn.CrossEntropyLoss()
    #TODO: see if defining criterion is necessary
    loss_ = criterion(pred, real) * mask
    return torch.mean(loss_)

### sort batch function to be able to use with pad_packed_sequence
def sort_batch(X, y, lengths):
    lengths, indx = lengths.sort(dim=0, descending=True)
    X = X[indx]
    if y is not None:
        y = y[indx]
    return X.transpose(0,1), y, lengths # transpose (batch x seq) to (seq x batch)


# Converts the unicode file to ascii
def unicode_to_ascii(s):
    """
    Normalizes latin chars with accent to their canonical decomposition
    """
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')

# Preprocessing the sentence to add the start, end tokens and make them lower-case
def preprocess_sentence_eng_esp(w):
    w = unicode_to_ascii(w.lower().strip())
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)

    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)

    w = w.rstrip().strip()
    w = ['<sos>'] + w.split(' ') + ['<eos>']
    return w

def get_tensor(name, example):
    return ['<sos>'] + list(example) + ['<eos>']

    # if name == 'out':
    #     vals = example.split("^")
    # else:
    #     vals = list(example)
    #
    # vals = ['<sos>'] + vals + ['<eos>']
    #
    # return vals

def max_length(tensor):
    return max(len(t) for t in tensor)

def pad_sequences(x, max_len):
    padded = np.zeros((max_len), dtype=np.int64)
    if len(x) > max_len: padded[:] = x[:max_len]
    else: padded[:len(x)] = x
    return padded


class Vocab_Lang():
    def __init__(self, data):
        """ data is the list of all sentences in the language dataset"""
        self.data = data
        self.word2idx = {}
        self.idx2word = {}
        self.vocab = set()

        self.create_index()

    def create_index(self):
        for sentence in self.data:
            self.vocab.update(sentence)

        # add a padding token
        self.word2idx['<pad>'] = 0
        self.word2idx['<sos>'] = 1
        self.word2idx['<eos>'] = 2

        # word to index mapping
        index = 3
        for _, word in enumerate(self.vocab):
            if word != '<sos>' and word != '<eos>':
                self.word2idx[word] = index   # +3 because of pad, sos, and eos tokens
                index +=1
           #     print(word)

        # index to word mapping
        for word, index in self.word2idx.items():
            self.idx2word[index] = word

    def matrix2words(self, matrix):
        sentences = []
        for i in range(matrix.shape[0]):
            words = []
            for j in range(matrix.shape[1]):
                if matrix[i,j].tolist() == self.word2idx['<pad>']:
                    break
                words.append(self.idx2word[matrix[i,j].tolist()])
            sentences.append(' '.join(words))
        return sentences


# conver the data to tensors and pass to the Dataloader
# to create an batch iterator


class MyData(Dataset):
    def __init__(self, X, y):
        self.data = X
        self.target = y
        # TODO: convert this into torch code if possible
        self.length = [np.sum(1 - np.equal(x, 0)) for x in X]

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        x_len = self.length[index]
        return x, y, x_len

    def __len__(self):
        return len(self.data)



def get_reference_candidate(target, pred):
  reference = list(target)
  reference = [trg_lang.idx2word[s] for s in np.array(reference[1:])]
  candidate = list(pred)
  candidate = [trg_lang.idx2word[s] for s in np.array(candidate[1:])]
  return reference, candidate


def evaluate(trg,pred):
    assert pred.shape[0] == trg.shape[0]
    num_examples = pred.shape[0]

    # accuracy
    num_same = 0
    for i in range(pred.shape[0]):
        pred_row = [x for x in pred[i, :].tolist() if x != 0]
        trg_row = [x for x in trg[i, :].tolist() if x != 0]
        if pred_row == trg_row:
            num_same += 1
    acc = num_same / num_examples

    # unrestricted accuracy
    num_same = 0
    for i in range(pred.shape[0]):
        pred_row = [x for x in pred[i,:].tolist() if x != 0]
        trg_row = [x for x in trg[i,:].tolist() if x != 0]
        if sorted(pred_row) == sorted(trg_row):
            num_same += 1
    unrestricted_acc = num_same / num_examples

    # edit distance
    edit_dists = [None] * pred.shape[0]
    for i in range(pred.shape[0]):
        pred_row = [x for x in pred[i, :].tolist() if x != 0]
        trg_row = [x for x in trg[i, :].tolist() if x != 0]
        edit_dists[i] = edit_distance(pred_row, trg_row)

    min_dist = min(edit_dists)
    max_dist = max(edit_dists)
    avg_dist = sum(edit_dists) / len(edit_dists)

    return (acc, unrestricted_acc, min_dist, avg_dist, max_dist)

def edit_distance(str1, str2):
    '''Simple Levenshtein implementation for evalm.'''
    table = np.zeros([len(str2) + 1, len(str1) + 1])
    for i in range(1, len(str2) + 1):
        table[i][0] = table[i - 1][0] + 1
    for j in range(1, len(str1) + 1):
        table[0][j] = table[0][j - 1] + 1
    for i in range(1, len(str2) + 1):
        for j in range(1, len(str1) + 1):
            if str1[j - 1] == str2[i - 1]:
                dg = 0
            else:
                dg = 1
            table[i][j] = min(table[i - 1][j] + 1, table[i][j - 1] + 1,
                              table[i - 1][j - 1] + dg)
    return int(table[len(str2)][len(str1)])

def train_model(trn_dataset, dev_dataset, model):
    print(torch.cuda.is_available())

    ## Feel free to change any parameters class definitions as long as you can change the training code, but make sure
    ## evaluation should get the tensor format it expects, this is only for reference
    n_batch = len(trn_dataset.dataset) // model.batch_sz

    optimizer = optim.Adam(list(model.encoder.parameters()) + list(model.decoder.parameters()),
                           lr=0.001)

    for epoch in range(EPOCHS):
        start = time.time()

        model.encoder.train()
        model.decoder.train()

        total_loss = 0

        for (batch, (inp, targ, inp_len)) in enumerate(trn_dataset):
            loss = 0
            inp, targ, inp_len = inp, targ, inp_len  # batch_size x MAX_LEN_SRC; batch_size x MAX_LEN_TAR; batch_size
            xs, ys, lens = sort_batch(inp, targ,
                                      inp_len)  # MAX_LEN_SRC x batch_size; batch_size x MAX_LEN_TAR; batch_size

            enc_output, enc_hidden = model.encoder(xs.to(DEVICE),
                                             lens)  # , device=device)   # MAX_LEN_SRC x batch_size x hidden_size; batch_size x hidden_size
            dec_hidden = enc_hidden  # batch_size x hidden_size

            # use teacher forcing - feeding the target as the next input (via dec_input)
            dec_input = torch.tensor([[trg_lang.word2idx['<sos>']]] * model.batch_sz)  # batch_size x 1
            # run code below for every timestep in the ys batch
            for t in range(1, ys.size(1)):
                predictions, dec_hidden, _ = model.decoder(dec_input.to(DEVICE),  # batch_size x 1
                                                     dec_hidden.to(DEVICE),  # batch_size x hidden_size
                                                     enc_output.to(DEVICE))  # MAX_LEN_SRC x batch_size x hidden_size
                # batch_size x vocab_size; batch_size x 1024; batch_size x 1 x MAX_LEN_SRC

                loss += loss_function(ys[:, t].to(DEVICE), predictions.to(DEVICE))
                # loss += loss_
                dec_input = ys[:, t].unsqueeze(1)

            batch_loss = (loss / int(ys.size(1)))
            total_loss += float(batch_loss)

            optimizer.zero_grad()

            loss.backward()

            ### UPDATE MODEL PARAMETERS
            optimizer.step()

            if batch % 100 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                             batch,
                                                             batch_loss.detach().item()))


        ### TODO: Save checkpoint for model
        print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                            total_loss / n_batch))
        pred_output = inference_of_model(model, dev_dataset)
        eval = evaluate(torch.Tensor(dev_dataset.dataset.target),pred_output)

        print("Eval metrics at epoch {}: ".format(epoch + 1), eval)
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
    return epoch

def inference_of_model(model, dataset):
    start = time.time()

    model.encoder.eval()
    model.decoder.eval()

    total_loss = 0

    pred_output = torch.zeros((len(dataset.dataset), MAX_TRG_LENGTH))

    for (batch, (inp, _, inp_len)) in enumerate(dataset):
        #TODO: implement dev/test loss
        #loss = 0
        xs, _, lens = sort_batch(inp, None, inp_len)
        enc_output, enc_hidden = model.encoder(xs.to(DEVICE), lens)
        dec_hidden = enc_hidden

        dec_input = torch.tensor([[trg_lang.word2idx['<sos>']]] * xs.size(1))
        eos = torch.tensor([[trg_lang.word2idx['<sos>']]] * xs.size(1)) # all ones, zeros mean eos has been predicted

        curr_output = torch.zeros((xs.size(1), MAX_TRG_LENGTH)) #ys.size(1)))
        curr_output[:, 0] = dec_input.squeeze(1)

        for t in range(1, MAX_TRG_LENGTH):  # run code below for every timestep in the ys batch
            predictions, dec_hidden, _ = model.decoder(dec_input.to(DEVICE),
                                                 dec_hidden.to(DEVICE),
                                                 enc_output.to(DEVICE))
            #loss += loss_function(ys[:, t].to(DEVICE), predictions.to(DEVICE))
            eos = eos.to(DEVICE)
            dec_input = torch.argmax(predictions, dim=1).unsqueeze(1)

            # update eos
            dec_input = dec_input * eos # zero out any predictions where eos has already been predicted,
                                        # should just be dec_input the first time through

            # update eos to include <eos> just predicted
            tmp0 = dec_input.clone()    # don't want to change original tensor
            tmp0 = (tmp0!=2) * 1        # any end of sentence predictions turn into 0s
            eos = eos * tmp0            # which in turn zeros out ones in eos

            curr_output[:, t] = dec_input.squeeze(1)

        pred_output[batch * model.batch_sz: (batch + 1) * xs.size(1)] = curr_output
        #batch_loss = (loss / int(ys.size(1)))
        #total_loss += float(batch_loss)
    # print('Epoch {} Loss {:.4f}'.format(epoch + 1,
    #                                     total_loss / N_BATCH))
    print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
    # print('Accuracy: {}'.format(evaluate(target_output, pred_output)))

    return pred_output #, target_output



def prep_pairs(file, lang):
    print(file)
    f = open(file, encoding='UTF-8').read().strip().split('\n')
    lines = f
    if lang == 'esp':
        total_num_examples = 30000
    else:
        total_num_examples = len(lines)
    original_word_pairs = [[w for w in l.split('\t')][:2] for l in lines[:total_num_examples]]
    return original_word_pairs

def pairs2panda(data_pairs):
    data = pd.DataFrame(data_pairs, columns=["inp", "out"])
    data['inp'] = data.inp.apply(lambda w: get_tensor('inp', w))
    data['out'] = data.out.apply(lambda w: get_tensor('out', w))
    return data

def prepare_data(data, inp_lang, trg_lang):
    # Vectorize the input and target languages for the data
    input_tensor = [[inp_lang.word2idx[s] for s in inp] for inp in data["inp"].values.tolist()]
    target_tensor = [[trg_lang.word2idx[s] for s in out] for out in data["out"].values.tolist()]

    # calculate the max_length of input and output tensor for padding
    max_length_inp, max_length_tar = max_length(input_tensor), max_length(target_tensor)
    # pad all the sentences in the dataset with the max_length
    input_tensor = [pad_sequences(x, max_length_inp) for x in input_tensor]
    target_tensor = [pad_sequences(x, max_length_tar) for x in target_tensor]

    return input_tensor, target_tensor

if __name__ == "__main__":
    import sys
    print(torch.__version__)
    args = configure_training(sys.argv[1:])

    # enables switching between esp-eng to ess
    lang = args.lang

    data_file = args.data
    EPOCHS = args.num_epochs

    # open file and extract word pairs for train, dev, and test
    if lang == 'esp':
        original_word_pairs = prep_pairs(data_file, lang)
    elif lang == 'ess':
        original_word_pairs_trn = prep_pairs(data_file + 'train', lang)
        original_word_pairs_dev = prep_pairs(data_file + 'dev', lang)
        original_word_pairs_tst = prep_pairs(data_file + 'test', lang)


    if args.lang == 'esp':
        # preprocessing using pandas and lambdas
        data = pd.DataFrame(original_word_pairs, columns=["out", "inp"])
        data["out"] = data.out.apply(lambda w: preprocess_sentence_eng_esp(w))
        data["inp"] = data.inp.apply(lambda w: preprocess_sentence_eng_esp(w))
        data_trn = data[:24000]
        data_dev = data[24000:27000]
        data_tst = data[27000:]
        print(args.lang)
        #print(data[250:260])
    elif args.lang == 'ess':
        # returns pandas with columns 'inp' and 'out'
        data_trn = pairs2panda(original_word_pairs_trn)
        data_dev = pairs2panda(original_word_pairs_dev)
        data_tst = pairs2panda(original_word_pairs_tst)
        print(args.lang)
        data = pd.concat([data_trn, data_dev, data_tst])



    # create vocab from training data
    #TODO: URGENT make this be computed only from train data
    inp_lang = Vocab_Lang(data["inp"].values.tolist())
    trg_lang = Vocab_Lang(data["out"].values.tolist())

    input_tensor_trn, target_tensor_trn = prepare_data(data_trn, inp_lang, trg_lang)
    input_tensor_dev, target_tensor_dev = prepare_data(data_dev, inp_lang, trg_lang)
    input_tensor_tst, target_tensor_tst = prepare_data(data_tst, inp_lang, trg_lang)


    # BUFFER_SIZE = len(input_tensor_trn)
    batch_size = 60
    # N_BATCH = BUFFER_SIZE//BATCH_SIZE
    embedding_dim = 256

    # hidden size CHANGE THIS
    units = 1024
    vocab_inp_size = len(inp_lang.word2idx)
    vocab_tar_size = len(trg_lang.word2idx)

    model = Model(vocab_inp_size, vocab_tar_size, embedding_dim, units, batch_size)

    train_dataset = MyData(input_tensor_trn, target_tensor_trn)
    dev_dataset = MyData(input_tensor_dev, target_tensor_dev)
    tst_dataset = MyData(input_tensor_tst, target_tensor_tst)

    trn_dataset = DataLoader(train_dataset, batch_size = batch_size,
                         drop_last=True,
                         shuffle=True)

    dev_dataset = DataLoader(dev_dataset, batch_size = batch_size,
                             drop_last=True,
                             shuffle=False)

    tst_dataset = DataLoader(tst_dataset, batch_size=batch_size,
                             drop_last=True,
                             shuffle=False)

    # Device
    print(data_trn['inp'])
    print(data_trn['out'])

    epoch = train_model(trn_dataset, dev_dataset, model)
    pred_output = inference_of_model(model, tst_dataset)
    eval = evaluate(torch.Tensor(tst_dataset.dataset.target),pred_output)

    print("test set eval metrics: ", eval)

    if lang == 'esp':
        bleu_1 = 0.0
        bleu_2 = 0.0
        bleu_3 = 0.0
        bleu_4 = 0.0
        smoother = SmoothingFunction()
        save_candidate = []

        for i in range(len(tst_dataset.dataset)):
            reference, candidate = get_reference_candidate(tst_dataset.dataset.target[i], pred_output[i])
            # print(reference)
            # print(candidate)
            save_candidate.append(candidate)

            bleu_1 += sentence_bleu(reference, candidate, weights=(1, 0, 0, 0), smoothing_function=smoother.method1)
            bleu_2 += sentence_bleu(reference, candidate, weights=(0, 1, 0, 0), smoothing_function=smoother.method2)
            bleu_3 += sentence_bleu(reference, candidate, weights=(0, 0, 1, 0), smoothing_function=smoother.method3)
            bleu_4 += sentence_bleu(reference, candidate, weights=(0, 0, 0, 1), smoothing_function=smoother.method4)

        print('Individual 1-gram: %f' % (bleu_1 / len(target_tensor_dev)))
        print('Individual 2-gram: %f' % (bleu_2 / len(target_tensor_dev)))
        print('Individual 3-gram: %f' % (bleu_3 / len(target_tensor_dev)))
        print('Individual 4-gram: %f' % (bleu_4 / len(target_tensor_dev)))
        assert (len(save_candidate) == len(target_tensor_dev))

