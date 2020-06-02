import pickle
from collections import Counter
from utils import showPlot
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import random
import time
import math


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = '../data/kftt-data-1.0/data/orig/'
teacher_forcing_ratio = 0.5
BATCH_SIZE = 1
MAX_LENGTH = 50
dw = 300
dh = 256
SOS_token = 0
EOS_token = 1


with open('%skyoto-%s.ja.tokens' % (DATA_DIR, 'dev'), 'rb') as f:
    ja = pickle.load(f)

with open('%skyoto-%s.en.tokens' % (DATA_DIR, 'dev'), 'rb') as f:
    en = pickle.load(f)


'''
1. enとjaそれぞれword2idとid2wordを作る : 2回以上出てくる単語のみidを振り，それ以外は2にする．
2. 日本語が10単語以上の文は削除する．
3. tokensをidに変える．文頭と文末に0(sos), 1(eos)を追加する
4. lengsを保存する
5. paddingする
'''


def get_word2id(tokens, start=3, unk=2):  # input: list of tokens
    tokens = sum(tokens, [])  # flat list
    counter = Counter(tokens)
    word2id = {}
    for index, (token, freq) in enumerate(counter.most_common(), start):
        if freq < 2:
            word2id[token] = unk
        else:
            word2id[token] = index
    return word2id


def encode(tokens, word2id, sos=0, eos=1, unk=2):
    ids = [sos]
    for token in tokens:
        if token in word2id:
            ids.append(word2id[token])
        else:
            ids.append(unk)
    ids.append(eos)
    length = len(ids)
    return torch.tensor(ids).view(-1, 1), length


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


class Mydatasets(torch.utils.data.Dataset):
    def __init__(self, inputs, in_lengs, targets, target_lengs, max_len):
        inputs = inputs
        targets = targets
        self.inputs = inputs
        self.targets = targets
        # self.inputs = pad_sequence(inputs, batch_first=True)
        # self.targets = pad_sequence(targets, batch_first=True)
        self.in_lengs = in_lengs
        self.target_lengs = target_lengs
        self.data_size = len(inputs)

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        inputs = self.inputs[idx]
        targets = self.targets[idx]
        # in_lengs = self.in_lengs[idx]
        # target_lengs = self.target_lengs[idx]
        return inputs, targets


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer,
          decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    # encoder
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    # decoder
    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)

            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input
            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def trainIters(encoder, decoder, loader, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    criterion = nn.NLLLoss()

    for iter, (inputs, targets) in enumerate(loader, 1):
        input_tensor = inputs.squeeze(0)
        target_tensor = targets.squeeze(0)

        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)

        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    showPlot(plot_losses)


# bag-of-words にする．vocab_sizeを保存しておく
word2id_ja = get_word2id(ja)
ja = [encode(tokens, word2id_ja) for tokens in ja]
vocab_size_ja = len(word2id_ja)

word2id_en = get_word2id(en)
en = [encode(tokens, word2id_en) for tokens in en]
vocab_size_en = len(word2id_en)

X_train = []
X_lengs = []
Y_train = []
Y_lengs = []

# filter train data
for (ids_ja, len_ja), (ids_en, len_en) in zip(ja, en):
    if len_ja > 50 or len_en > 50:
        continue
    X_train.append(ids_ja)
    X_lengs.append(len_ja)
    Y_train.append(ids_en)
    Y_lengs.append(len_en)


trainset = Mydatasets(X_train, X_lengs, Y_train, Y_lengs, MAX_LENGTH)
loader = DataLoader(trainset, batch_size=BATCH_SIZE)

# for inputs, targets in loader:
#     print('inputs', inputs.squeeze(0))
#     print('targets', targets.squeeze(0)[0].size())
#     break


encoder = EncoderRNN(vocab_size_ja, dh)
decoder = AttnDecoderRNN(dh, vocab_size_en)

trainIters(encoder, decoder, loader, 10, print_every=2, plot_every=2, learning_rate=0.01)
