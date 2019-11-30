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
import pickle
from random import randint


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
parser.add_argument('--lr', type=float, default=4,
                    help='initial learning rate (default: 4)')
parser.add_argument('--nhid', type=int, default=600,
                    help='number of hidden units per layer (default: 600)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed (default: 1111)')
parser.add_argument('--tied', action='store_false',
                    help='tie the encoder-decoder weights (default: True)')
parser.add_argument('--optim', type=str, default='SGD',
                    help='optimizer type (default: SGD)')
parser.add_argument('--validseqlen', type=int, default=40,
                    help='valid sequence length (default: 40)')
parser.add_argument('--seq_len', type=int, default=80,
                    help='total sequence length, including effective history (default: 80)')
parser.add_argument('--corpus', action='store_true',
                    help='force re-make the corpus (default: False)')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
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
model = TCN(args.emsize, 1, num_chans, dropout=dropout, kernel_size=k_size)

if args.cuda:
    model.cuda()

# May use adaptive softmax to speed up training
criterion = nn.BCELoss()

lr = args.lr
optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr)


def evaluate(data_source):
    model.eval()
    total_loss = 0
    processed_data_size = 0
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(data_source):
            data = data.view(1, data.size(0), data.size(1))
            label = label.view(1, label.size(0))
            if args.cuda:
                data = data.cuda()
                label = label.cuda()
            output = model(data)

            loss = criterion(output, label)

            total_loss += loss.item()
            processed_data_size += data.size(1)
        return total_loss / processed_data_size


def train():
    # Turn on training mode which enables dropout.
    global train_data
    model.train()
    total_loss = 0
    start_time = time.time()
    for batch_idx, (data, label) in enumerate(train_data):
        data = data.view(1, data.size(0), data.size(1))
        label = label.view(1, label.size(0))
        if args.cuda:
            data = data.cuda()
            label = label.cuda()
        optimizer.zero_grad()
        output = model(data)

        loss = criterion(output, label)

        loss.backward()
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % args.log_interval == 0 and batch_idx > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.5f} | ms/batch {:5.5f} | '
                  'loss {:5.2f}'.format(
                epoch, batch_idx, len(train_data), lr,
                elapsed * 1000 / args.log_interval, cur_loss))
            total_loss = 0
            start_time = time.time()


if __name__ == "__main__":
    best_vloss = 1e8

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        all_vloss = []
        for epoch in range(1, args.epochs+1):
            epoch_start_time = time.time()
            train()
            val_loss = evaluate(valid_data)
            test_loss = evaluate(test_data)

            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                  .format(epoch, (time.time() - epoch_start_time),
                                               val_loss))
            print('| end of epoch {:3d} | time: {:5.2f}s | test loss {:5.2f} | '
                  .format(epoch, (time.time() - epoch_start_time),
                                            test_loss))
            print('-' * 89)

            # Save the model if the validation loss is the best we've seen so far.
            if val_loss < best_vloss:
                with open("model.pt", 'wb') as f:
                    print('Save model!\n')
                    torch.save(model, f)
                best_vloss = val_loss

            # Anneal the learning rate if the validation loss plateaus
            if epoch > 5 and val_loss >= max(all_vloss[-5:]):
                lr = lr / 2.
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            all_vloss.append(val_loss)

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    # Load the best saved model.
    with open("model.pt", 'rb') as f:
        model = torch.load(f)

    # Run on test data.
    test_loss = evaluate(test_data)
    print('=' * 89)
    print('| End of training | test loss {:5.2f}'.format(
        test_loss))
    print('=' * 89)
