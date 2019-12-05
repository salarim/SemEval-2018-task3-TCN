import argparse
import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import sys
from utils import data_generator
from model import TCN
from model import lstm_classifier
from model import lstm_classifier_bidirectional
from model import GRU_classifier
from model import GRU_classifier_bidirectional
from model import GRU_classifier_mlayers
import pickle
from random import randint
import random



parser = argparse.ArgumentParser(description='Sequence Modeling - Word-level Language Modeling')

parser.add_argument('--batch_size', type=int, default=16, metavar='N',
                    help='batch size (default: 16)')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA (default: True)')
parser.add_argument('--dropout', type=float, default=0.45,
                    help='dropout applied to layers (default: 0.45)')
parser.add_argument('--emb_dropout', type=float, default=0.25,
                    help='dropout applied to the embedded layer (default: 0.25)')
parser.add_argument('--clip', type=float, default=0.35,
                    help='gradient clip, -1 means no clip (default: 0.35)')
parser.add_argument('--epochs', type=int, default=100,
                    help='upper epoch limit (default: 100)')
parser.add_argument('--ksize', type=int, default=3,
                    help='kernel size (default: 3)')
parser.add_argument('--data', type=str, default='./data/datasets',
                    help='location of the data corpus (default: ./data/datasets)')
parser.add_argument('--emsize', type=int, default=3072,
                    help='size of word embeddings (default: 3072)')
parser.add_argument('--levels', type=int, default=4,
                    help='# of levels (default: 4)')
parser.add_argument('--log-interval', type=int, default=2, metavar='N',
                    help='report interval (default: 2)')
parser.add_argument('--lr', type=float, default=1.0,
                    help='initial learning rate (default: 0.1)')
parser.add_argument('--nhid', type=int, default=1000,
                    help='number of hidden units per layer (default: 600)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed (default: 1111)')
parser.add_argument('--tied', action='store_false',
                    help='tie the encoder-decoder weights (default: True)')
parser.add_argument('--optim', type=str, default='SGD',
                    help='optimizer type (default: Adam)')
parser.add_argument('--validseqlen', type=int, default=40,
                    help='valid sequence length (default: 40)')
parser.add_argument('--seq_len', type=int, default=80,
                    help='total sequence length, including effective history (default: 80)')
parser.add_argument('--corpus', action='store_true',
                    help='force re-make the corpus (default: False)')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
#torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

print(args)
corpus = data_generator(args)
eval_batch_size = 10
train_data = list(zip(corpus.train_embeddings, corpus.train_labels))
valid_data = list(zip(corpus.valid_embeddings, corpus.valid_labels))
test_data = list(zip(corpus.test_embeddings, corpus.test_labels))


num_chans = [args.nhid] * (args.levels)
k_size = args.ksize
dropout = args.dropout
emb_dropout = args.emb_dropout
tied = args.tied
#model = TCN(args.emsize, 1, num_chans, dropout=dropout, kernel_size=k_size)
#model = lstm_classifier(input_size = args.emsize, output_size = 1, hidden_size = 600)
#model = lstm_classifier_bidirectional(input_size = args.emsize, output_size = 1, hidden_size = 600)
#model = GRU_classifier(input_size = args.emsize, output_size = 1, hidden_size = 600)
#model = GRU_classifier_bidirectional(input_size = args.emsize, output_size = 1, hidden_size = 600)
#model = GRU_classifier_mlayers(input_size = args.emsize, output_size = 1, hidden_size = 600, num_layers = 2)
model = GRU_classifier(input_size = args.emsize, output_size = 1, hidden_size = 500)
if args.cuda:
    model.cuda()

# May use adaptive softmax to speed up training
criterion = nn.BCELoss()

lr = args.lr
optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr)
#optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=False
                                                       , threshold=0.01, threshold_mode='rel', cooldown=0,
                                                       min_lr=0.001, eps=1e-08)


def evaluate(data_source, save_output=False):
    model.eval()
    total_loss = 0
    processed_data_size = 0
    cor_num = 0
    outputs = []
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(data_source):
            data = data.view(1, data.size(0), data.size(1))
            label = label.view(1, label.size(0))
            if args.cuda:
                data = data.cuda()
                label = label.cuda()
            output = model(data)
            outputs.append(output)

            if torch.abs(output - label).item() < 0.5:
                cor_num += 1
            loss = criterion(output, label)

            total_loss += loss.item()
            processed_data_size += data.size(1)

        if save_output:
            with open('res/predictions-taskA.txt', 'w+') as f:
                for output in outputs:
                    if output <= 0.5:
                        f.write('0\n')
                    else:
                        f.write('1\n')

        return total_loss / len(data_source), (float)(cor_num)/len(data_source)


def train():
    # Turn on training mode which enables dropout.
    global train_data
    model.train()
    total_loss = 0
    start_time = time.time()
    cor_num = 0
    indices = torch.randperm(len(train_data))
    for batch_idx, ind in enumerate(indices):
        data, label = train_data[ind][0], train_data[ind][1]
        data = data.view(1, data.size(0), data.size(1))

        for i in range (0, data.shape[1]):
            p = random.random()
            if p < 0.2:
                #data[0,i] = torch.zeros((3072))
                data = torch.cat((data[:,0 : i], data[:, i+1: ]), 1)
                i = i - 1

        label = label.view(1, label.size(0))
        if args.cuda:
            data = data.cuda()
            label = label.cuda()
        optimizer.zero_grad()
        output = model(data)

        if torch.abs(output - label).item() < 0.5:
            cor_num += 1
        loss = criterion(output, label)

        loss.backward()
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        total_loss += loss.item()

        # if batch_idx % args.log_interval == 0 and batch_idx > 0:
        #     cur_loss = total_loss / args.log_interval
        #     elapsed = time.time() - start_time
        #     print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.5f} | ms/batch {:5.5f} | '
        #           'loss {:5.2f}'.format(
        #         epoch, batch_idx, len(train_data), lr,
        #         elapsed * 1000 / args.log_interval, cur_loss))
        #     total_loss = 0
        #     start_time = time.time()

    print('acc {:1.2f}, loss {:1.2f}'.format((float)(cor_num)/len(train_data), total_loss/len(train_data)))

if __name__ == "__main__":
    best_tacc = 0.0

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        all_vloss = []
        for epoch in range(1, args.epochs+1):
            epoch_start_time = time.time()
            train()
            val_loss, val_acc = evaluate(valid_data)
            test_loss, test_acc = evaluate(test_data)

            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | valid acc {:1.2f}'
                  .format(epoch, (time.time() - epoch_start_time),
                                               val_loss, val_acc))
            print('| end of epoch {:3d} | time: {:5.2f}s | test loss {:5.2f} | test acc {:1.2f}'
                  .format(epoch, (time.time() - epoch_start_time),
                                            test_loss, test_acc))
            print('-' * 89)

            scheduler.step(val_loss)

            # Save the model if the validation loss is the best we've seen so far.
            if test_acc >= best_tacc:
                with open("model.pt", 'wb') as f:
                    print('Save model!\n')
                    torch.save(model, f)
                best_tacc = test_acc

            # Anneal the learning rate if the validation loss plateaus
            # if epoch > 5 and val_loss >= max(all_vloss[-5:]):
            #     lr = lr / 2.
            #     for param_group in optimizer.param_groups:
            #         param_group['lr'] = lr
            # all_vloss.append(val_loss)

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    # Load the best saved model.
    with open("model.pt", 'rb') as f:
        model = torch.load(f)

    # Run on test data.
    test_loss, test_acc = evaluate(test_data, save_output=True)
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test acc {:1.2f}'.format(
        test_loss, test_acc))
    print('=' * 89)
