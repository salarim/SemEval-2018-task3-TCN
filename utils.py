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
        corpus = Corpus(args.data,
                        '/train/SemEval2018-T3-train-taskA.txt',
                        '/goldtest_TaskA/SemEval2018-T3_gold_test_taskA_emoji.txt',
                        '/test_TaskA/SemEval2018-T3_input_test_taskA.txt')
        pickle.dump(corpus, open(args.data + '/corpus', 'wb'))
    return corpus


class Corpus:
    def __init__(self, path, trainfile, testfile_emoji, testfile_without_emoji):
        self.train_embeddings, self.train_labels = self.create_embeddings(path, trainfile, trainfile)
        self.test_embeddings, self.test_labels = self.create_embeddings(path, testfile_emoji, testfile_without_emoji, is_test=True)
        self.valid_embeddings = self.test_embeddings[:int(len(self.test_embeddings)*0.1)]
        self.valid_labels = self.test_labels[:int(len(self.test_labels)*0.1)]

    def create_embeddings(self, path, file_emoji, file_witout_emoji, is_test=False):
        embeddings = []
        labels = []
        break_point = 10
        with open(path + file_witout_emoji, encoding="utf8") as fp:
            lines = fp.readlines()
            lines = lines[1:len(lines)]
            for i, l in enumerate(lines):
                if i > break_point:
                    break
                line = l.split("\t")
                sent_index = 1 if is_test else 2
                sentence = line[sent_index]
                sentence = pre_process(sentence)
                sent_embed = get_bert_embedding(sentence)
                sent_embed = torch.stack(sent_embed)
                embeddings.append(sent_embed)

        with open(path + file_emoji, encoding="utf8") as fp:
            lines = fp.readlines()
            lines = lines[1:len(lines)]
            for i, l in enumerate(lines):
                if i > break_point:
                    break
                line = l.split("\t")
                labels.append(torch.FloatTensor([float(line[1])]))

        return embeddings, labels

