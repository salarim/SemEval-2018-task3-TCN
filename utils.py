import os
import torch
from torch.autograd import Variable
import pickle
from tokenizer import get_bert_embedding, pre_process
import re

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



class GloveCorpus:
    FLAGS = re.MULTILINE | re.DOTALL

    def __init__(self):
        self.glove_embeddings = self.load_glove_embedding()
        print(self.glove_embeddings['<user>'])
        pass

    def create_embeddings(self, path, file_emoji, is_test=False):
        embeddings = []
        labels = []
        break_point = 10
        with open(path + file_emoji, encoding="utf8") as fp:
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
                labels.append(torch.FloatTensor([float(line[1])]))

    def hashtag(self, text):
        text = text.group()
        hashtag_body = text[1:]
        if hashtag_body.isupper():
            result = "<hashtag> {} <allcaps>".format(hashtag_body)
        else:
            result = " ".join(["<hashtag>"] + re.split(r"(?=[A-Z])", hashtag_body, flags=FLAGS))
        return result

    def allcaps(self, text):
        text = text.group()
        return text.lower() + " <allcaps>"

    def tokenize(self, text):
        # Different regex parts for smiley faces
        eyes = r"[8:=;]"
        nose = r"['`\-]?"

         # function so code less repetitive
        def re_sub(pattern, repl):
            return re.sub(pattern, repl, text, flags=FLAGS)

        text = re_sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", "<url>")
        text = re_sub(r"/"," / ")
        text = re_sub(r"@\w+", "<user>")
        text = re_sub(r"{}{}[)dD]+|[)dD]+{}{}".format(eyes, nose, nose, eyes), "<smile>")
        text = re_sub(r"{}{}p+".format(eyes, nose), "<lolface>")
        text = re_sub(r"{}{}\(+|\)+{}{}".format(eyes, nose, nose, eyes), "<sadface>")
        text = re_sub(r"{}{}[\/|l*]".format(eyes, nose), "<neutralface>")
        text = re_sub(r"<3","<heart>")
        text = re_sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", "<number>")
        text = re_sub(r"#\S+", self.hashtag)
        text = re_sub(r"([!?.]){2,}", r"\1 <repeat>")
        text = re_sub(r"\b(\S*?)(.)\2{2,}\b", r"\1\2 <elong>")

        ## -- I just don't understand why the Ruby script adds <allcaps> to everything so I limited the selection.
        # text = re_sub(r"([^a-z0-9()<>'`\-]){2,}", allcaps)
        text = re_sub(r"([A-Z]){2,}", self.allcaps)

        return text.lower()

    def load_glove_embedding(self):
        glove_embeddings = {}
        with open('data/embeddings/glove.twitter.27B/glove.twitter.27B.200d.txt') as f:
            for l in f:
                tokens = l.split()
                glove_embeddings[tokens[0]] = torch.FloatTensor([float(x) for x in tokens[1:]])
        return glove_embeddings





