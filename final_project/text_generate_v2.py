
# coding: utf-8

##############################################################################
#language Modeling on Penn Tree Bank
#
# This file generates new sentences sampled from the language model
#
###############################################################################

import argparse

import torch
from torch.autograd import Variable

import data_v3
import json
import pandas as pd
import preprocess
import os
from sklearn.model_selection import ShuffleSplit
import random
import numpy as np

seed = random.seed(20180330)


parser = argparse.ArgumentParser(description='PyTorch bbc Language Model')

# Model parameters.
parser.add_argument('--data', type=str, default='./data/',
                    help='location of the data corpus')
parser.add_argument('--checkpoint', type=str, default='./model.pt',
                    help='model checkpoint to use')
parser.add_argument('--outf', type=str, default='generated.txt',
                    help='number of output file for generated text')
parser.add_argument('--words', type=int, default='1000',
                    help='words to generate')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--temperature', type=float, default=1.0,
                    help='temperature - higher will increase diversity')
parser.add_argument('--log-interval', type=int, default=100,
                    help='reporting interval')
args = parser.parse_args()


# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

if args.temperature < 1e-3:
    parser.error("--temperature has to be greater or equal 1e-3")

with open(args.checkpoint, 'rb') as f:
    model = torch.load(f)
model.eval()

if args.cuda:
    model.cuda()
else:
    model.cpu()


vocab = preprocess.read_vocab(os.path.join(args.data,'VOCAB.txt'))
idx_train = pd.read_json('idx_train.json')
idx_val = pd.read_json('idx_val.json')
idx_test = pd.read_json('idx_test.json')

# Load pretrained Embeddings
gn_glove_dir = './gn_glove/1b-vectors300-0.8-0.8.txt' #142527 tokens, last one is '<unk>'
ntokens = sum(1 for line in open(gn_glove_dir)) + 1

with open(gn_glove_dir) as f: 
    gn_glove_vecs = np.zeros((ntokens, 300))
    words2idx_emb = {}
    idx2words_emb = []
    # ordered_words = []
    for i, line in enumerate(f):
        try:
            s = line.split() 
            gn_glove_vecs[i, :] = np.asarray(s[1:])
            words2idx_emb[s[0]] = i
            idx2words_emb.append(s[0])
            # ordered_words.append(s[0])
        except:
            continue

    words2idx_emb['<eos>'] = i+1
    idx2words_emb.append('<eos>')
    gn_glove_vecs[i+1, :] = np.random.normal(size=300)

# Load data
corpus = data_v3.Corpus('./data/outputdata', vocab, words2idx_emb, idx2words_emb, idx_train, idx_test, idx_val) #改动2 
ntokens = len(corpus.idx2words)
hidden = model.init_hidden(1)
input = Variable(torch.rand(1, 1).mul(ntokens).long(), volatile=True)
if args.cuda:
    input.data = input.data.cuda()

with open(args.outf, 'w') as outf:
    for i in range(args.words):
        output, hidden = model(input, hidden)
        word_weights = output.squeeze().data.div(args.temperature).exp().cpu()
        word_idx = torch.multinomial(word_weights, 1)[0]
        input.data.fill_(word_idx)
        word = corpus.idx2words[word_idx]

        outf.write(word + ('\n' if i % 20 == 25 else ' '))

        if i % args.log_interval == 0:
            print('| Generated {}/{} words'.format(i, args.words))
