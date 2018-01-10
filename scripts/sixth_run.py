from __future__ import unicode_literals, print_function, division
import pandas as pd
import csv
import numpy as np

from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

import time
import math

SOS_token = 0
EOS_token = 1
MAX_LENGTH = 200
teacher_forcing_ratio = 0.5
model_name = 'sixth_run_downsample'

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        for i in range(self.n_layers):
            output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1):
        super(DecoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax()

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        for i in range(self.n_layers):
            output = F.relu(output)
            output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_output, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)))
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        for i in range(self.n_layers):
            output = F.relu(output)
            output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]))
        return output, hidden, attn_weights

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result

check_trans = lambda s: (len(s) == len(s.encode())) and ('.' not in s) and ('www' not in s) and ('http' not in s)

def filterPair(p):
    return len(p[0]) < MAX_LENGTH

def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

def getChars(item):
    return [element for element in item]

def get_data(file_range):
    print('File range:', list(file_range))
    data_list = []
    for i in file_range:
        index = str(i)
        if len(index) == 1:
            filename = 'output-0000{}-of-00100'.format(index)
        elif len(index) == 2:
            filename = 'output-000{}-of-00100'.format(index)
        else:
            raise ValueError('Wrong index')

        cur_data = pd.read_csv('../input/ru_with_types/' + filename, sep='\t', names=['class', 'before', 'after'],
                           quoting=csv.QUOTE_NONE, encoding='utf-8', dtype=str, na_filter=False)

        is_trans = cur_data.before.astype(str).apply(lambda x: check_trans(x))
        cur_data.loc[(is_trans) & (cur_data['class'] == 'PLAIN'), 'class'] = 'TRANS'

        if (cur_data.shape[0] > 1074563-10) and (cur_data.shape[0] < 1074563+10):
            print(filename)
        data_list.append(cur_data)
        print('Data shape for item {} is {}'.format(i,cur_data.shape))


    data_orig = pd.concat(data_list, axis=0)
    print('Overall data shape is {}'.format(data_orig.shape))

    return data_orig

def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

def variableFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    result = Variable(torch.LongTensor(indexes).view(-1, 1))
    if use_cuda:
        return result.cuda()
    else:
        return result


def variablesFromPair(pair):
    input_variable = variableFromSentence(input_lang, pair[0])
    target_variable = variableFromSentence(output_lang, pair[1])
    return (input_variable, target_variable)

def make_sample(data_learn, self_frac = 0.33, sil_frac = 1):

    data_nn = data_learn.copy()
    to_concat = []
    to_concat.append(data_nn[(data_nn.after != '<self>') & (data_nn.after != 'sil')])
    to_concat.append(data_nn[data_nn.after == '<self>'].sample(frac = self_frac))
    to_concat.append(data_nn[data_nn.after == 'sil'].sample(frac = sil_frac))

    data_nn = pd.concat(to_concat, axis=0)
    return data_nn

def get_pairs(data_orig, filter_length = MAX_LENGTH, downsample_common = 500):

    big_str = list(data_orig.before.astype(str).values)
    output_list = list(data_orig.after.astype(str).values)
    types_list = list(data_orig['class'].values)

    grp = data_orig.fillna('NaN').groupby(by='before')
    group_sizes = grp.size().to_dict()

    stride = 3
    input_list = []
    pairs = []
    for i in range(len(big_str)):
        if big_str[i] != '<eos>':
            #print(big_str[i])
            cur_item = ['<norm>'] + getChars(big_str[i]) + ['</norm>']
            cur_type = types_list[i]
            cur_item = ['<{}>'.format(cur_type)] + cur_item + ['</{}>'.format(cur_type)]
            #print(big_str[i-stride:i])
            prefix = getChars(' '.join(big_str[i-stride:i]))
            #print(prefix)
            prefix = ' '.join(prefix).split('< e o s >')[-1].split(' ')
            #print(prefix)
            suffix = getChars(' '.join(big_str[i+1:i+stride+1]))
            suffix = ' '.join(suffix).split('< e o s >')[0].split(' ')
            cur_item = prefix \
            + cur_item + \
            suffix

            cur_item = ' '.join(cur_item)
            cur_item = cur_item.replace('  ', ' ')
            cur_item = cur_item.replace('  ', ' ')
            if cur_item[0] == ' ':
                cur_item = cur_item[1:]
            if downsample_common:
                #print(big_str[i])
                leave_flag = np.random.random(1)[0] < downsample_common/group_sizes[big_str[i]]
            else:
                leave_flag = True
            pairs += [(cur_item, output_list[i], cur_type, leave_flag)]


    #pairs = list(zip(input_list, output_list))
    print('Len of pairs:', len(pairs))

    if filter_length:
        pairs = filterPairs(pairs)
        print('Len of pairs after filtering:', len(pairs))
    return pairs

def train(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer,
          criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_variable.size()[0]
    target_length = target_variable.size()[0]

    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_variable[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0][0]

    decoder_input = Variable(torch.LongTensor([[SOS_token]]))
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_output, encoder_outputs)
            loss += criterion(decoder_output, target_variable[di])
            decoder_input = target_variable[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_output, encoder_outputs)
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]

            decoder_input = Variable(torch.LongTensor([[ni]]))
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input

            loss += criterion(decoder_output, target_variable[di])
            if ni == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0] / target_length

def trainIters_weighted(encoder, decoder, pairs, test_pairs,
               n_iters, print_every=1000, plot_every=100,
               learning_rate=0.01, evaluate_each=False, min_class_size = 100, add_weighted = False):
    start = time.time()
    plot_losses = []
    plot_accuracies = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    criterion = nn.NLLLoss()

    classes = ['PLAIN', 'PUNCT', 'VERBATIM', 'ORDINAL', 'MEASURE', 'DATE',
           'ELECTRONIC', 'CARDINAL', 'LETTERS', 'DECIMAL', 'FRACTION',
           'TELEPHONE', 'TIME', 'MONEY', 'DIGIT', 'TRANS', '<eos>']

    initial_weights = [3.0 for i in range(len(classes) -1)] + [0.0]
    initial_errors = [0.5 for i in range(len(classes) -1)] + [0.0]
    weight_dict = dict(zip(classes, initial_weights))
    weight_dict["PLAIN"] = 1
    weight_dict["PUNCT"] = 2

    error_dict = dict(zip(classes, initial_errors))
    cur_iter = 1
    epoch_lag = 20
    for big_iter in range(1, int(np.ceil(n_iters/evaluate_each))):

        if add_weighted:
            even_sample = make_even_sample(pairs, size_of_class = min_class_size)
            weighted_sample = sample_pairs(pairs, size = evaluate_each - len(even_sample) + 1,
                                           weight_dict = weight_dict)
            sample = weighted_sample + even_sample
        else:
            num_classes = (len(classes) - 1)
            class_size = int(evaluate_each/num_classes)
            sample = make_even_sample(pairs, size_of_class = class_size)
            if num_classes*class_size < evaluate_each:
                weighted_sample = sample_pairs(pairs, size = evaluate_each - num_classes*class_size,
                                           weight_dict = weight_dict)
                sample += weighted_sample

        print(sample[0])
        random.shuffle(sample)
        print(sample[0])
        print(len(sample))
        training_pairs = [variablesFromPair(item)
                      for item in sample]

        for iter in range(1, evaluate_each + 1):
            training_pair = training_pairs[iter - 1]
            input_variable = training_pair[0]
            target_variable = training_pair[1]

            loss = train(input_variable, target_variable, encoder,
                         decoder, encoder_optimizer, decoder_optimizer, criterion)
            print_loss_total += loss
            plot_loss_total += loss

            if cur_iter % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (timeSince(start, cur_iter / n_iters),
                                             cur_iter, cur_iter / n_iters * 100, print_loss_avg))
            cur_iter += 1

            if iter % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0

            if evaluate_each and iter % evaluate_each == 0 and iter != 0:

                cur_accuracy, new_error_dict = evaluate_pairs(encoder, decoder, test_pairs)
                for item in classes[:-1]:
                    if new_error_dict[item] >= error_dict[item] and new_error_dict[item] > 0.05:
                        weight_dict[item] += 1
                    else:
                        error_dict[item] = new_error_dict[item]
                #error_dict = new_error_dict
                #weight_dict['<eos>'] = 0.0
                print(weight_dict)
                plot_accuracies.append(cur_accuracy)
            '''
            if plot_losses and np.min(plot_losses) not in plot_losses[-epoch_lag:]:
                learning_rate = learning_rate/np.sqrt(10)
                epoch_lag += 5
                print('Setting new learning rate to {:.5f}'.format(learning_rate))
                encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
                decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
            '''

    #showPlot(plot_losses)
    #showPlot(plot_accuracies)

    return plot_losses

def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    input_variable = variableFromSentence(input_lang, sentence)
    input_length = input_variable.size()[0]
    encoder_hidden = encoder.initHidden()

    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_variable[ei],
                                                 encoder_hidden)
        encoder_outputs[ei] = encoder_outputs[ei] + encoder_output[0][0]

    decoder_input = Variable(torch.LongTensor([[SOS_token]]))  # SOS
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = encoder_hidden

    decoded_words = []
    decoder_attentions = torch.zeros(max_length, max_length)

    for di in range(max_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_output, encoder_outputs)
        decoder_attentions[di] = decoder_attention.data
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        if ni == EOS_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(output_lang.index2word[ni])

        decoder_input = Variable(torch.LongTensor([[ni]]))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    return decoded_words, decoder_attentions[:di + 1]

def sample_pairs(train_pairs, size = 1000, weight_dict = None):

    classes = ['PLAIN', 'PUNCT', 'VERBATIM', 'ORDINAL', 'MEASURE', 'DATE',
           'ELECTRONIC', 'CARDINAL', 'LETTERS', 'DECIMAL', 'FRACTION',
           'TELEPHONE', 'TIME', 'MONEY', 'DIGIT', 'TRANS', '<eos>']

    if weight_dict is None:
        weights = [1 for i in range(len(classes) -1)] + [0.0]
        weight_dict = dict(zip(classes, weights))

        weight_dict['PLAIN'] = 0.05
        weight_dict['PUNCT'] = 0.15
        weight_dict['DECIMAL'] = 5
        weight_dict['FRACTION'] = 5
        weight_dict['MONEY'] = 20
        weight_dict['TIME'] = 10
        weight_dict['ELECTRONIC'] = 10
        weight_dict['ELECTRONIC'] = 10
        weight_dict['DIGIT'] = 10


    sample_weights = np.array([weight_dict[item[2]] for item in train_pairs])
    sample_weights = sample_weights/ sample_weights.sum()

    sample_indices = np.random.choice(range(len(train_pairs)), size = size, p=sample_weights)
    sample = [train_pairs[i] for i in sample_indices]

    return sample

def make_even_sample(pairs, size_of_class = 100):

    classes = ['PLAIN', 'PUNCT', 'VERBATIM', 'ORDINAL', 'MEASURE', 'DATE',
           'ELECTRONIC', 'CARDINAL', 'LETTERS', 'DECIMAL', 'FRACTION',
           'TELEPHONE', 'TIME', 'MONEY', 'DIGIT', 'TRANS']

    sample = []
    for item in classes:
        class_pairs = [pair for pair in pairs if pair[2] == item]
        sample_indices = np.random.choice(range(len(class_pairs)), size = size_of_class)
        cur_sample = [class_pairs[i] for i in sample_indices]
        sample += cur_sample

    return sample

def evaluate_pairs(encoder, decoder, test_pairs):

    classes = ['PLAIN', 'PUNCT', 'VERBATIM', 'ORDINAL', 'MEASURE', 'DATE',
           'ELECTRONIC', 'CARDINAL', 'LETTERS', 'DECIMAL', 'FRACTION',
           'TELEPHONE', 'TIME', 'MONEY', 'DIGIT', 'TRANS']

    results_dict = dict.fromkeys(classes)
    preds = np.array([(item[1], ' '.join(evaluate(encoder, decoder, item[0])[0][:-1]), item[0]) for item in test_pairs])
    results = np.array([item[0] == item[1] for item in preds])
    print('\t\t eval accuracy: {:.3f}'.format(results.mean()))

    for item in classes:
        results_dict[item] = 1 - np.mean([results[i] for i in range(len(results)) if test_pairs[i][2] == item])
        print('\t\t\t {} eval error: {:.3f}'.format(item, results_dict[item]))

    results_dict['<eos>'] = 0
    return results.mean(), results_dict

def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

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

#setting seeds
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

use_cuda = torch.cuda.is_available()

data_dev = get_data(range(0,2))
data_learn = get_data(range(5,20))

train_pairs = get_pairs(data_learn, downsample_common = 1000)
dev_pairs = get_pairs(data_dev, downsample_common = False)

input_lang, output_lang = Lang('nonnorm'), Lang('norm')

print('Filtering common words in train...')
train_pairs = [item for item in train_pairs if item[3]]
print('Done. Updated length is {}'.format(len(train_pairs)))

for pair in train_pairs + dev_pairs:
    input_lang.addSentence(pair[0])
    output_lang.addSentence(pair[1])

print(train_pairs[:5])
print(dev_pairs[:5])

torch.backends.cudnn.enabled = False
test_pairs = make_even_sample(dev_pairs, size_of_class = 400)

test_weight = dict((data_dev['class'].value_counts()/len(data_dev)))
print(test_weight)
hidden_size = 256
encoder1 = EncoderRNN(input_lang.n_words, hidden_size, n_layers=5)
attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words,
                           n_layers = 3, dropout_p=0.2)

if use_cuda:
    print('Using CUDA')
    encoder1 = encoder1.cuda()
    attn_decoder1 = attn_decoder1.cuda()

callback_num = 1000

num_iterations = 200000

print('LR 0.01')
plot_losses = trainIters_weighted(encoder1, attn_decoder1, train_pairs, test_pairs, num_iterations, print_every=callback_num,
                         plot_every=callback_num, evaluate_each=20000, learning_rate = 0.01,
                        min_class_size = 800, add_weighted = False)

print('LR 0.001')
plot_losses = trainIters_weighted(encoder1, attn_decoder1, train_pairs, test_pairs, num_iterations, print_every=callback_num,
                         plot_every=callback_num, evaluate_each=20000, learning_rate = 0.001,
                        min_class_size = 800, add_weighted = False)

print('LR 0.0001')
plot_losses = trainIters_weighted(encoder1, attn_decoder1, train_pairs, test_pairs, 2*num_iterations, print_every=callback_num,
                         plot_every=callback_num, evaluate_each=20000, learning_rate = 0.0001,
                        min_class_size = 800, add_weighted = False)



torch.save(encoder1.state_dict() , 'models/encoder_{}_iters.states'.format(model_name))
torch.save(attn_decoder1.state_dict(), 'models/decoder_{}_iters.states'.format(model_name))

print('Done.')
