import os
import torch
from torch.autograd import Variable
import pickle
from tokenizer import get_bert_embedding, pre_process

"""
Note: The meaning of batch_size in PTB is different from that in MNIST example. In MNIST, 
batch_size is the # of sample data that is considered in each iteration; in PTB, however,
it is the number of segments to speed up computation. 

The goal of PTB is to train a language model to predict the next word.
"""


def data_generator(args):
    if os.path.exists(args.data + "/corpus") and not args.corpus:
        corpus = pickle.load(open(args.data + '/corpus', 'rb'))
    else:
        corpus = Corpus(args.data, '/train/SemEval2018-T3-train-taskA.txt')
        pickle.dump(corpus, open(args.data + '/corpus', 'wb'))
    return corpus


class Corpus:
    def __init__(self, path, filename):
        embeddings = []
        labels = []
        with open(path + filename, encoding="utf8") as fp:
            lines = fp.readlines()
            lines = lines [1:len(lines)]
            for i, line in enumerate(lines):
                if i > 10:
                    break
                line = line.split("\t")
                sentence = line[2]
                labels.append(torch.FloatTensor([float(line[1])]))
                sentence = pre_process(sentence)
                sent_embed = get_bert_embedding(sentence)
                sent_embed = torch.stack(sent_embed)
                embeddings.append(sent_embed)

        train_perc, valid_perc, test_perc = 0.7, 0.2, 0.1
        self.train_embeddings = embeddings[:int(len(embeddings)*train_perc)]
        self.valid_embeddings = embeddings[int(len(embeddings)*train_perc):int(len(embeddings)*(train_perc+valid_perc))]
        self.test_embeddings = embeddings[int(len(embeddings)*(train_perc+valid_perc)):int(len(embeddings)*(train_perc+valid_perc+test_perc))]

        self.train_labels = labels[:int(len(labels)*train_perc)]
        self.valid_labels = labels[int(len(labels)*train_perc):int(len(labels)*(train_perc+valid_perc))]
        self.test_labels = labels[int(len(labels)*(train_perc+valid_perc)):int(len(labels)*(train_perc+valid_perc+test_perc))]
